package ai.rai;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.io.StringWriter;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

    UnitTypeTable utt;
    int maxAttackDiameter;

    static Process pythonProcess;
    static BufferedReader inPipe;
    static DataOutputStream outPipe;

    boolean sentInitialMapInformation;

    public RAISocketAI(UnitTypeTable a_utt) {
        super(100, -1);
        utt = a_utt;
        maxAttackDiameter = utt.getMaxAttackRange() * 2 + 1;
        try {
            connectChildProcess();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public RAISocketAI(int mt, int mi, UnitTypeTable a_utt) {
        super(mt, mi);
        utt = a_utt;
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
        ProcessBuilder processBuilder = new ProcessBuilder("rai_microrts");
        pythonProcess = processBuilder.start();
        inPipe = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream()));
        outPipe = new DataOutputStream(pythonProcess.getOutputStream());

        reset();
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

    @Override
    public void reset() {
        try {
            sentInitialMapInformation = false;

            StringWriter sw = new StringWriter();
            utt.toJSON(sw);
            send(RAISocketMessageType.UTT, new byte[][] { sw.toString().getBytes(StandardCharsets.UTF_8) });

            if (DEBUG >= 1)
                System.out.println("RAISocketAI: UTT sent, waiting for ack");

            // wait for ack:
            inPipe.readLine();

            // read any extra left-over lines
            while (inPipe.ready())
                inPipe.readLine();
            if (DEBUG >= 1)
                System.out.println("RAISocketAI: ack received");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        GameStateWrapper gsw = new GameStateWrapper(gs, DEBUG);

        Gson gson = new Gson();

        if (DEBUG >= 1)
            System.out.println("RAISocketAI: getAction sending");

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

        send(RAISocketMessageType.GET_ACTION, obs.toArray(new byte[0][]));

        if (DEBUG >= 1)
            System.out.println("RAISocketAI: getAction sent, waiting for actions");

        String actionString = inPipe.readLine();
        Type int2d = new TypeToken<int[][]>() {
        }.getType();
        int[][] actionVector = gson.fromJson(actionString, int2d);
        PlayerAction pa = PlayerAction.fromVectorAction(actionVector, gs, utt, player, maxAttackDiameter);
        pa.fillWithNones(gs, player, 1);

        if (DEBUG >= 1)
            System.out.println("RAISocketAI: actions received");

        return pa;
    }

    @Override
    public void preGameAnalysis(GameState gs, long milliseconds, String readWriteFolder) throws Exception {
        GameStateWrapper gsw = new GameStateWrapper(gs, DEBUG);
        PhysicalGameState pgs = gs.getPhysicalGameState();

        if (DEBUG >= 1)
            System.out.println("RAISocketAI: preGameAnalysis sending");

        ArrayList<byte[]> obs = new ArrayList<>(Arrays.asList(
                gsw.getArrayObservation(0),
                gsw.getBinaryMask(0),
                gsw.getPlayerResources(0),
                new byte[] { (byte) pgs.getHeight(), (byte) pgs.getWidth() },
                gsw.getTerrain()));
        if (DEBUG >= 1) {
            Gson gson = new Gson();
            obs.add(gson.toJson(gsw.getVectorObservation(0)).getBytes(StandardCharsets.UTF_8));
            obs.add(gson.toJson(gsw.getMasks(0)).getBytes(StandardCharsets.UTF_8));
        }

        send(RAISocketMessageType.PRE_GAME_ANALYSIS, obs.toArray(new byte[0][]));
        sentInitialMapInformation = true;

        inPipe.readLine();
    }

    @Override
    public void gameOver(int winner) throws Exception {
        send(RAISocketMessageType.GAME_OVER, new byte[][] { new byte[] { (byte) winner } });

        // wait for ack:
        inPipe.readLine();
    }

    @Override
    public AI clone() {
        if (DEBUG >= 1)
            System.out.println("RAISocketAI: cloning");
        return new RAISocketAI(TIME_BUDGET, ITERATIONS_BUDGET, utt);
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        List<ParameterSpecification> l = new ArrayList<>();

        return l;
    }
}
