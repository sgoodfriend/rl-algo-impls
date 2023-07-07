package ai.rai;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringWriter;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import ai.core.AI;
import ai.core.AIWithComputationBudget;
import ai.core.ParameterSpecification;
import rts.GameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.units.UnitTypeTable;

public class RAISocketAI extends AIWithComputationBudget {
    public static int DEBUG;
    public int PYTHON_VERBOSE_LEVEL = 1;
    public int OVERRIDE_TORCH_THREADS = 0;

    UnitTypeTable utt;
    int maxAttackDiameter;

    static Process pythonProcess;
    static BufferedReader inPipe;
    static DataOutputStream outPipe;
    static ThreadPoolExecutor executor;
    static Future<String> pendingRequestHandler;

    boolean sentInitialMapInformation;

    public RAISocketAI(UnitTypeTable a_utt) {
        this(100, -1, a_utt, 0, 1);
    }

    public RAISocketAI(int mt, int mi, UnitTypeTable a_utt) {
        this(mt, mi, a_utt, 0, 1);
    }

    public RAISocketAI(int mt, int mi, UnitTypeTable a_utt, int overrideTorchThreads, int pythonVerboseLevel) {
        super(mt, mi);
        utt = a_utt;
        OVERRIDE_TORCH_THREADS = overrideTorchThreads;
        PYTHON_VERBOSE_LEVEL = pythonVerboseLevel;
        maxAttackDiameter = utt.getMaxAttackRange() * 2 + 1;
        try {
            connectChildProcess();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void connectChildProcess() throws Exception {
        if (pythonProcess != null) {
            return;
        }
        List<String> command = new ArrayList<>(Arrays.asList(
                "rai_microrts",
                "--time_budget_ms",
                String.valueOf(TIME_BUDGET),
                "--override_torch_threads",
                String.valueOf(OVERRIDE_TORCH_THREADS)));
        if (PYTHON_VERBOSE_LEVEL > 0) {
            command.add("-" + "v".repeat(PYTHON_VERBOSE_LEVEL));
        }
        ProcessBuilder processBuilder = new ProcessBuilder();
        processBuilder.command(command);
        pythonProcess = processBuilder.start();

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            pythonProcess.destroy();
        }));

        inPipe = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()));
        outPipe = new DataOutputStream(pythonProcess.getOutputStream());

        executor = new ThreadPoolExecutor(
                2, 4,
                5000, TimeUnit.MILLISECONDS,
                new LinkedBlockingDeque<Runnable>());

        reset();
    }

    private void pauseChildProcess() {
        if (pythonProcess == null) {
            return;
        }
        if (DEBUG >= 1) {
            System.out.println("RAISocketAI: Pausing Python process");
        }
        try {
            new ProcessBuilder("kill", "-STOP", String.valueOf(pythonProcess.pid())).start();
        } catch (IOException e) {
            if (DEBUG >= 1) {
                e.printStackTrace();
            }
        }
    }

    private void resumeChildProcess() {
        if (pythonProcess == null) {
            return;
        }
        if (DEBUG >= 1) {
            System.out.println("RAISocketAI: Resuming Python process");
        }
        try {
            new ProcessBuilder("kill", "-CONT", String.valueOf(pythonProcess.pid())).start();
        } catch (IOException e) {
            if (DEBUG >= 1) {
                e.printStackTrace();
            }
        }
    }

    public void send(RAISocketMessageType messageType, byte[][] bs) throws Exception {
        int sz = (2 + bs.length) * 4 + Arrays.stream(bs).mapToInt(b -> b.length).sum();
        ByteBuffer bb = ByteBuffer.allocate(sz);
        bb.putInt(messageType.ordinal());
        bb.putInt(bs.length);
        for (byte[] b : bs) {
            bb.putInt(b.length);
        }
        for (byte[] b : bs) {
            bb.put(b);
        }
        send(bb.array());
    }

    public void send(byte[] b) throws Exception {
        outPipe.writeInt(b.length);
        outPipe.write(b);
        outPipe.flush();
    }

    public String request(RAISocketMessageType messageType, byte[][] bs) throws Exception {
        return request(messageType, bs, null);
    }

    public String request(RAISocketMessageType messageType, byte[][] bs, Long timeoutMillis) throws Exception {
        long startTime = System.currentTimeMillis();
        if (pendingRequestHandler != null) {
            resumeChildProcess();
            try {
                if (timeoutMillis != null) {
                    pendingRequestHandler.get(timeoutMillis, TimeUnit.MILLISECONDS);
                } else {
                    pendingRequestHandler.get();
                }
                pendingRequestHandler = null;
            } catch (TimeoutException e) {
                if (DEBUG >= 1) {
                    System.out.println("RAISocketAI: Prior request exceeded new timeout!");
                }
                pauseChildProcess();
                return null;
            } catch (InterruptedException | ExecutionException e) {
                if (DEBUG >= 1) {
                    System.out.println("RAISocketAI: Prior request errored:");
                    e.printStackTrace();
                }
                pendingRequestHandler = null;
            }
            if (timeoutMillis != null) {
                timeoutMillis -= System.currentTimeMillis() - startTime;
                if (DEBUG >= 1) {
                    System.out.println("RAISocketAI: Time remaining " + timeoutMillis);
                }
                if (timeoutMillis <= 0) {
                    return null;
                }
            }
        }
        send(messageType, bs);
        if (DEBUG >= 2) {
            System.out.println("RAISocketAI: sent " + messageType.name());
        }
        Callable<String> task = () -> {
            var response = inPipe.readLine();
            if (DEBUG >= 2) {
                System.out.println("RAISocketAI: received response to " + messageType.name());
            }
            return response;
        };
        if (timeoutMillis == null) {
            return task.call();
        }
        pendingRequestHandler = executor.submit(task);
        try {
            var response = pendingRequestHandler.get(timeoutMillis, TimeUnit.MILLISECONDS);
            pendingRequestHandler = null;
            return response;
        } catch (TimeoutException e) {
            if (DEBUG >= 1) {
                System.out.println("RAISocketAI: Request timed out");
            }
            pauseChildProcess();
            return null;
        } catch (InterruptedException | ExecutionException e) {
            if (DEBUG >= 1) {
                System.out.println("RAISocketAI: Exception thrown");
                e.printStackTrace();
            }
            pendingRequestHandler = null;
            return null;
        }
    }

    @Override
    public void reset() {
        try {
            sentInitialMapInformation = false;

            StringWriter sw = new StringWriter();
            utt.toJSON(sw);
            request(RAISocketMessageType.UTT, new byte[][] { sw.toString().getBytes(StandardCharsets.UTF_8) });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        long startTime = System.currentTimeMillis();
        GameStateWrapper gsw = new GameStateWrapper(gs, DEBUG);

        Gson gson = new Gson();

        ArrayList<byte[]> obs = new ArrayList<>(Arrays.asList(
                gsw.getArrayObservation(player),
                gsw.getBinaryMask(player),
                gsw.getPlayerResources(player)));
        if (!sentInitialMapInformation || DEBUG >= 1) {
            sentInitialMapInformation = true;
            PhysicalGameState pgs = gs.getPhysicalGameState();
            obs.add(new byte[] { (byte) pgs.getHeight(), (byte) pgs.getWidth() });
            obs.add(gsw.getTerrain());
            if (DEBUG >= 1) {
                obs.add(gson.toJson(gsw.getVectorObservation(player)).getBytes(StandardCharsets.UTF_8));
                obs.add(gson.toJson(gsw.getMasks(player)).getBytes(StandardCharsets.UTF_8));
            }
        }
        long timeoutMillis = TIME_BUDGET - (System.currentTimeMillis() - startTime);
        if (DEBUG >= 2) {
            System.out.println("RAISocketAI: Remaining time budget: " + timeoutMillis);
        }
        var response = request(RAISocketMessageType.GET_ACTION, obs.toArray(new byte[0][]),
                Long.valueOf(timeoutMillis));
        PlayerAction pa;
        if (response != null) {
            Type int2d = new TypeToken<int[][]>() {
            }.getType();
            int[][] actionVector = gson.fromJson(response, int2d);
            pa = PlayerAction.fromVectorAction(actionVector, gs, utt, player, maxAttackDiameter);
        } else {
            System.out.println("RAISocketAI: Empty getAction response (likely timeout). Returning empty action");
            pa = new PlayerAction();
        }
        pa.fillWithNones(gs, player, 1);

        return pa;
    }

    @Override
    public void preGameAnalysis(GameState gs, long milliseconds, String readWriteFolder) throws Exception {
        GameStateWrapper gsw = new GameStateWrapper(gs, DEBUG);
        PhysicalGameState pgs = gs.getPhysicalGameState();

        ArrayList<byte[]> obs = new ArrayList<>(Arrays.asList(
                gsw.getArrayObservation(0),
                gsw.getBinaryMask(0),
                gsw.getPlayerResources(0),
                new byte[] { (byte) pgs.getHeight(), (byte) pgs.getWidth() },
                gsw.getTerrain(),
                ByteBuffer.allocate(8).putLong(milliseconds).array(),
                readWriteFolder.getBytes(StandardCharsets.UTF_8)));
        if (DEBUG >= 1) {
            Gson gson = new Gson();
            obs.add(gson.toJson(gsw.getVectorObservation(0)).getBytes(StandardCharsets.UTF_8));
            obs.add(gson.toJson(gsw.getMasks(0)).getBytes(StandardCharsets.UTF_8));
        }

        request(RAISocketMessageType.PRE_GAME_ANALYSIS, obs.toArray(new byte[0][]));
        sentInitialMapInformation = true;
    }

    @Override
    public void gameOver(int winner) throws Exception {
        request(RAISocketMessageType.GAME_OVER, new byte[][] { new byte[] { (byte) winner } });
    }

    @Override
    public AI clone() {
        if (DEBUG >= 1)
            System.out.println("RAISocketAI: cloning");
        return new RAISocketAI(TIME_BUDGET, ITERATIONS_BUDGET, utt, OVERRIDE_TORCH_THREADS, PYTHON_VERBOSE_LEVEL);
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        List<ParameterSpecification> l = new ArrayList<>();

        return l;
    }
}
