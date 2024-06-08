package tests;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.StringWriter;
import java.io.Writer;
import java.nio.file.Paths;

import ai.core.AI;
import ai.jni.JNIAI;
import ai.jni.Response;
import ai.rai.RAIBotResponse;
import ai.rai.GameStateWrapper;
import ai.reward.RewardFunctionInterface;
import gui.PhysicalGameStateJFrame;
import gui.PhysicalGameStatePanel;
import rts.GameState;
import rts.PartiallyObservableGameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.TraceEntry;
import rts.units.Unit;
import rts.units.UnitTypeTable;

/**
 * Instances of this class each let us run a single environment (or sequence
 * of them, if we reset() in between) between two players.
 * 
 * In this client, it is assumed that actions are selected by Java-based bots
 * for **both** players. See RAIGridnetClient.java for a client where one
 * player is externally controlled, and RAIGridnetClientSelfPlay.java for one
 * where both players are externally controlled.
 *
 * @author santi and costa
 */
public class RAIBotClient {

    PhysicalGameStateJFrame w;
    public AI ais[];
    public PhysicalGameState pgs;
    public GameState gs;
    public GameState[] playergs = new GameState[2];
    UnitTypeTable utt;
    boolean partialObs;
    public RewardFunctionInterface[] rfs;
    public String mapPath;
    String micrortsPath;
    boolean gameover = false;
    boolean layerJSON = true;
    public int renderTheme = PhysicalGameStatePanel.COLORSCHEME_WHITE;
    public int maxAttackRadius;
    public final int numPlayers = 2;

    // storage
    int[][][][] masks = new int[2][][][];
    double[][] rewards = new double[2][];
    boolean[][] dones = new boolean[2][];
    RAIBotResponse[] response = new RAIBotResponse[2];
    PlayerAction[] pas = new PlayerAction[2];

    /**
     * 
     * @param a_rfs          Reward functions we want to use to compute rewards at
     *                       every step.
     * @param a_micrortsPath Path for the microrts root directory (with Java code
     *                       and maps).
     * @param a_mapPath      Path (under microrts root dir) for map to load.
     * @param a_ai1
     * @param a_ai2
     * @param a_utt
     * @param partial_obs
     * @throws Exception
     */
    public RAIBotClient(RewardFunctionInterface[] a_rfs, String a_micrortsPath, String a_mapPath, AI a_ai1, AI a_ai2,
            UnitTypeTable a_utt, boolean partial_obs) throws Exception {
        micrortsPath = a_micrortsPath;
        mapPath = a_mapPath;
        rfs = a_rfs;
        utt = a_utt;
        partialObs = partial_obs;
        if (a_ai1 == null || a_ai2 == null) {
            throw new Exception("no ai1 or ai2 was chosen");
        }
        if (micrortsPath.length() != 0) {
            this.mapPath = Paths.get(micrortsPath, mapPath).toString();
        }

        pgs = PhysicalGameState.load(mapPath, utt);

        ais = new AI[] { a_ai1, a_ai2 };
        // initialize storage
        for (int i = 0; i < numPlayers; i++) {
            masks[i] = new int[pgs.getHeight()][pgs.getWidth()][1 + 6 + 4 + 4 + 4 + 4 + utt.getUnitTypes().size()
                    + maxAttackRadius * maxAttackRadius];
            rewards[i] = new double[rfs.length];
            dones[i] = new boolean[rfs.length];
            response[i] = new RAIBotResponse(null, null, null, null, null, null, null, null);
        }
    }

    public byte[] render(boolean returnPixels) throws Exception {
        if (w == null) {
            w = PhysicalGameStatePanel.newVisualizer(gs, 640, 640, false, null, renderTheme);
        }
        w.setStateCloning(gs);
        w.repaint();

        if (!returnPixels) {
            return null;
        }
        BufferedImage image = new BufferedImage(w.getWidth(),
                w.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
        w.paint(image.getGraphics());

        WritableRaster raster = image.getRaster();
        DataBufferByte data = (DataBufferByte) raster.getDataBuffer();
        return data.getData();
    }

    public void gameStep() throws Exception {
        TraceEntry te = new TraceEntry(gs.getPhysicalGameState().clone(), gs.getTime());
        var errored = false;
        int[][][] vectorAction = new int[numPlayers][][];
        for (int i = 0; i < numPlayers; ++i) {
            playergs[i] = gs;
            if (partialObs) {
                playergs[i] = new PartiallyObservableGameState(gs, i);
            }
            try {
                pas[i] = ais[i].getAction(i, playergs[i]);
            } catch (Throwable e) {
                System.err.println("Player " + i + " Error: " + e.getMessage());
                e.printStackTrace();

                final var p = i;
                // Remove all AI units, which should cause it to auto-lose.
                var unitsToRemove = gs.getUnits().stream().filter(u -> u.getPlayer() == p).toArray(Unit[]::new);
                for (Unit u : unitsToRemove) {
                    gs.removeUnit(u);
                }
                errored = true;
                break;
            }
            gs.issueSafe(pas[i]);
            te.addPlayerAction(pas[i].clone());
            vectorAction[i] = GameStateWrapper.toVectorAction(gs, pas[i]);
        }

        // simulate:
        if (!errored) {
            gameover = gs.cycle();
        }
        if (gameover || errored) {
            for (int i = 0; i < numPlayers; ++i) {
                ais[i].gameOver(gs.winner());
            }
        }

        for (int i = 0; i < numPlayers; i++) {
            for (int j = 0; j < rfs.length; j++) {
                rfs[j].computeReward(i, 1 - i, te, gs);
                rewards[i][j] = rfs[j].getReward();
                dones[i][j] = rfs[j].isDone();
            }
            var gsw = new GameStateWrapper(gs, 0, true);
            response[i].set(
                    gsw.getArrayObservation(i),
                    gsw.getBinaryMask(i),
                    rewards[i],
                    dones[i],
                    "{}",
                    null,
                    gsw.getPlayerResources(i),
                    vectorAction[i]);
        }
    }

    public String sendUTT() throws Exception {
        Writer w = new StringWriter();
        utt.toJSON(w);
        return w.toString(); // now it works fine
    }

    /**
     * Resets the environment.
     * 
     * @throws Exception
     */
    public void reset() throws Exception {
        pgs = PhysicalGameState.load(mapPath, utt);
        gs = new GameState(pgs, utt);

        for (int i = 0; i < numPlayers; i++) {
            playergs[i] = gs;
            if (partialObs) {
                playergs[i] = new PartiallyObservableGameState(gs, i);
            }

            ais[i] = ais[i].clone();
            ais[i].reset();

            for (int j = 0; j < rewards.length; j++) {
                rewards[i][j] = 0;
                dones[i][j] = false;
            }

            var gsw = new GameStateWrapper(gs, 0, true);
            response[i].set(
                    gsw.getArrayObservation(i),
                    gsw.getBinaryMask(i),
                    rewards[i],
                    dones[i],
                    "{}",
                    gsw.getTerrain(),
                    gsw.getPlayerResources(i),
                    null);
        }
    }

    public RAIBotResponse getResponse(int player) {
        return response[player];
    }

    public void close() throws Exception {
        if (w != null) {
            System.out.println(this.getClass().getSimpleName() + ": Not disposing frame. Resource Leak!");
            // w.dispose();
        }
    }
}
