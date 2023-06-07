package ai.rai;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringWriter;
import java.lang.reflect.Type;
import java.net.ConnectException;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

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

    String serverAddress = "127.0.0.1";
    int serverPort = 56241;

    static Process pythonProcess;
    static Socket socket;
    static BufferedReader in_pipe;
    static DataOutputStream out_pipe;

    public RAISocketAI(UnitTypeTable a_utt) {
        super(100, -1);
        utt = a_utt;
        maxAttackDiameter = utt.getMaxAttackRange() * 2 + 1;
        try {
            connectToServer(true);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public RAISocketAI(int mt, int mi, String a_sa, int a_port, UnitTypeTable a_utt) {
        super(mt, mi);
        serverAddress = a_sa;
        serverPort = a_port;
        utt = a_utt;
        maxAttackDiameter = utt.getMaxAttackRange() * 2 + 1;
        try {
            connectToServer(false);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private RAISocketAI(int mt, int mi, UnitTypeTable a_utt, Socket a_socket) {
        super(mt, mi);
        utt = a_utt;
        maxAttackDiameter = utt.getMaxAttackRange() * 2 + 1;
        try {
            socket = a_socket;
            in_pipe = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            out_pipe = new DataOutputStream(socket.getOutputStream());

            // Consume the initial welcoming messages from the server
            while (!in_pipe.ready())
                ;

            while (in_pipe.ready())
                in_pipe.readLine();

            reset();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Creates a RAISocketAI from an existing socket.
     *
     * @param mt     The time budget in milliseconds.
     * @param mi     The iterations budget in milliseconds
     * @param socket The socket the ai will communicate over.
     */
    public static RAISocketAI createFromExistingSocket(int mt, int mi, rts.units.UnitTypeTable a_utt,
            java.net.Socket socket) {
        return new RAISocketAI(mt, mi, a_utt, socket);
    }

    private void startPythonProcess(boolean isProcessServer) throws Exception {
        ProcessBuilder processBuilder = new ProcessBuilder("rai_microrts", String.valueOf(serverPort),
                String.valueOf(isProcessServer));
        pythonProcess = processBuilder.start();
    }

    public void connectToServer() throws Exception {
        if (socket == null) {
            // Make connection and initialize streams
            boolean connected = false;
            int sleepMs = 100;
            while (!connected) {
                try {
                    socket = new Socket(serverAddress, serverPort);
                    connected = true;
                } catch (ConnectException e) {
                    if (DEBUG >= 1)
                        System.out.printf("Waiting %dms for server to be ready%n", sleepMs);
                    Thread.sleep(sleepMs);
                    sleepMs += 100;
                }
            }
            in_pipe = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            out_pipe = new DataOutputStream(socket.getOutputStream());

            if (DEBUG >= 1) {
                System.out.println("RAISocketAI: waiting for in_pipe ready");
            }

            // Consume the initial welcoming messages from the server
            while (!in_pipe.ready())
                ;
            if (DEBUG >= 1) {
                System.out.println("RAISocketAI: comsuming initial messages");
            }
            while (in_pipe.ready())
                in_pipe.readLine();

            if (DEBUG >= 1)
                System.out.println("RAISocketAI: welcome message received");

            reset();
        }
    }

    public void connectToServer(boolean createServer) throws Exception {
        if (socket != null) {
            return;
        }
        if (createServer) {
            final int minEphemeralPort = 49152;
            final int maxEphemeralPort = 65535;
            final int maxAttempts = 20;

            ServerSocket serverSocket = null;
            for (int i = 0; serverSocket == null && i < maxAttempts; ++i) {
                serverPort = ThreadLocalRandom.current().nextInt(minEphemeralPort, maxEphemeralPort + 1);
                try {
                    serverSocket = new ServerSocket(serverPort);
                    if (DEBUG >= 1) {
                        System.out.printf("RAISocketAI: ServerSocket started on port %d%n", serverPort);
                    }
                } catch (IOException e) {
                    if (DEBUG >= 1) {
                        System.out.printf("RAISocketAI: Port %d busy. Trying another one...%n", serverPort);
                    }
                }
            }

            startPythonProcess(false);

            serverSocket.setSoTimeout(20000);
            socket = serverSocket.accept();

            if (DEBUG >= 1) {
                System.out.printf("RAISocketAI: Connected to server on port %d!%n", serverPort);
            }
        } else {
            startPythonProcess(true);

            boolean connected = false;
            int sleepMs = 100;
            while (!connected) {
                try {
                    socket = new Socket(serverAddress, serverPort);
                    connected = true;
                } catch (ConnectException e) {
                    if (DEBUG >= 1)
                        System.out.printf("Waiting %dms for server to be ready%n", sleepMs);
                    Thread.sleep(sleepMs);
                    sleepMs += 100;
                }
            }
        }

        in_pipe = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        out_pipe = new DataOutputStream(socket.getOutputStream());

        reset();
    }

    public void send(String s) throws Exception {
        byte[] b = s.getBytes(StandardCharsets.UTF_8);
        out_pipe.writeInt(b.length);
        out_pipe.write(b);
        out_pipe.flush();
    }

    @Override
    public void reset() {
        try {
            StringWriter sw = new StringWriter();
            sw.append("utt\n");
            utt.toJSON(sw);
            send(sw.toString());

            if (DEBUG >= 1)
                System.out.println("RAISocketAI: UTT sent, waiting for ack");

            // wait for ack:
            in_pipe.readLine();

            // read any extra left-over lines
            while (in_pipe.ready())
                in_pipe.readLine();
            if (DEBUG >= 1)
                System.out.println("RAISocketAI: ack received");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        GameStateWrapper gsw = new GameStateWrapper(gs);

        Gson gson = new Gson();

        StringWriter sw = new StringWriter();
        sw.append("getAction\n");
        sw.append(gson.toJson(gsw.getVectorObservation(player))).append("\n");
        sw.append(gson.toJson(gsw.getMasks(player))).append("\n");

        PhysicalGameState pgs = gs.getPhysicalGameState();
        Map<String, Integer> mapData = new HashMap<>();
        mapData.put("height", pgs.getHeight());
        mapData.put("width", pgs.getWidth());
        sw.append(gson.toJson(mapData));

        if (DEBUG >= 1)
            System.out.println("RAISocketAI: getAction sending");

        send(sw.toString());

        if (DEBUG >= 1)
            System.out.println("RAISocketAI: getAction sent, waiting for actions");

        String actionString = in_pipe.readLine();
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
    public void preGameAnalysis(GameState gs, long milliseconds, String readWriteFolder) {

    }

    @Override
    public void gameOver(int winner) throws Exception {
        StringWriter sw = new StringWriter();
        sw.append("gameOver\n").append(String.valueOf(winner));
        send(sw.toString());

        // wait for ack:
        in_pipe.readLine();
    }

    @Override
    public AI clone() {
        if (DEBUG >= 1)
            System.out.println("RAISocketAI: cloning");
        return new RAISocketAI(TIME_BUDGET, ITERATIONS_BUDGET, serverAddress, serverPort, utt);
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        List<ParameterSpecification> l = new ArrayList<>();

        l.add(new ParameterSpecification("Server Address", String.class, "127.0.0.1"));
        l.add(new ParameterSpecification("Server Port", Integer.class, 9898));

        return l;
    }
}
