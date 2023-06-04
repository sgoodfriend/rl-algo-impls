package ai.rai;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.io.StringWriter;
import java.lang.reflect.Type;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import ai.core.AI;
import ai.core.AIWithComputationBudget;
import ai.core.ParameterSpecification;
import rts.GameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.UnitAction;
import rts.UnitActionAssignment;
import rts.units.Unit;
import rts.units.UnitTypeTable;

public class RAISocketAI extends AIWithComputationBudget {
    public static int DEBUG;

    UnitTypeTable utt;
    int maxAttackDiameter;

    String serverAddress = "127.0.0.1";
    int serverPort = 9898;
    Socket socket;
    BufferedReader in_pipe;
    DataOutputStream out_pipe;

    public RAISocketAI(UnitTypeTable a_utt) {
        super(100, -1);
        utt = a_utt;
        maxAttackDiameter = utt.getMaxAttackRange() * 2 + 1;
        try {
            connectToServer();
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
            connectToServer();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private RAISocketAI(int mt, int mi, UnitTypeTable a_utt, Socket socket) {
        super(mt, mi);
        utt = a_utt;
        maxAttackDiameter = utt.getMaxAttackRange() * 2 + 1;
        try {
            this.socket = socket;
            in_pipe = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            out_pipe = new DataOutputStream(socket.getOutputStream());

            // Consume the initial welcoming messages from the server
            while (!in_pipe.ready())
                ;
            while (in_pipe.ready())
                in_pipe.readLine();

            if (DEBUG >= 1) {
                System.out.println("SocketAI: welcome message received");
            }
            reset();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Creates a SocketAI from an existing socket.
     *
     * @param mt     The time budget in milliseconds.
     * @param mi     The iterations budget in milliseconds
     * @param socket The socket the ai will communicate over.
     */
    public static RAISocketAI createFromExistingSocket(int mt, int mi, rts.units.UnitTypeTable a_utt,
            java.net.Socket socket) {
        return new RAISocketAI(mt, mi, a_utt, socket);
    }

    public void startPythonProcess() {
        ProcessBuilder processBuilder = new ProcessBuilder("python",
                "src/rl-algo-impls/rl-algo-impls/microrts/agent.py");
    }

    public void connectToServer() throws Exception {
        // Make connection and initialize streams
        socket = new Socket(serverAddress, serverPort);
        in_pipe = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        out_pipe = new DataOutputStream(socket.getOutputStream());

        // Consume the initial welcoming messages from the server
        while (!in_pipe.ready())
            ;
        while (in_pipe.ready())
            in_pipe.readLine();

        if (DEBUG >= 1)
            System.out.println("SocketAI: welcome message received");

        reset();
    }

    public void close() throws Exception {
        socket.close();
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
                System.out.println("SocketAI: UTT sent, waiting for ack");

            // wait for ack:
            in_pipe.readLine();

            // read any extra left-over lines
            while (in_pipe.ready())
                in_pipe.readLine();
            if (DEBUG >= 1)
                System.out.println("SocketAI: ack received");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        Gson gson = new Gson();

        StringWriter sw = new StringWriter();
        sw.append("getAction\n");
        sw.append(gson.toJson(gs.getVectorObservation(player))).append("\n");
        sw.append(gson.toJson(getMasks(player, gs))).append("\n");

        PhysicalGameState pgs = gs.getPhysicalGameState();
        Map<String, Integer> mapData = new HashMap<>();
        mapData.put("height", pgs.getHeight());
        mapData.put("width", pgs.getWidth());
        sw.append(gson.toJson(mapData));

        send(sw.toString());

        String actionString = in_pipe.readLine();
        Type int2d = new TypeToken<int[][]>() {
        }.getType();
        int[][] actionVector = gson.fromJson(actionString, int2d);
        PlayerAction pa = PlayerAction.fromVectorAction(actionVector, gs, utt, player, maxAttackDiameter);
        pa.fillWithNones(gs, player, 1);
        return pa;
    }

    private int[][][] getMasks(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();

        int[][][] masks = new int[pgs.getHeight()][pgs.getWidth()][1 + 6 + 4 + 4 + 4 + 4 + utt.getUnitTypes().size()
                + maxAttackDiameter * maxAttackDiameter];
        for (Unit u : pgs.getUnits()) {
            final UnitActionAssignment uaa = gs.getActionAssignment(u);
            if (u.getPlayer() == player && uaa == null) {
                masks[u.getY()][u.getX()][0] = 1;
                UnitAction.getValidActionArray(u, gs, utt, masks[u.getY()][u.getX()], maxAttackDiameter, 1);
            }
        }

        return masks;
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

        close();
    }

    @Override
    public AI clone() {
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
