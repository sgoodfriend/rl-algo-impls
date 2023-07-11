/*
* To change this template, choose Tools | Templates
* and open the template in the editor.
*/
package tests;

import java.io.Writer;
import java.nio.file.Paths;
import java.awt.image.BufferedImage;
import java.io.StringWriter;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;

import ai.core.AI;
import ai.jni.JNIAI;
import ai.reward.RewardFunctionInterface;
import ai.jni.JNIInterface;
import ai.rai.RAIResponse;
import ai.rai.GameStateWrapper;
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
 *
 * @author santi
 * 
 *         Once you have the server running (for example, run
 *         "RunServerExample.java"), set the proper IP and port in the variable
 *         below, and run this file. One of the AIs (ai1) is run remotely using
 *         the server.
 * 
 *         Notice that as many AIs as needed can connect to the same server. For
 *         example, uncomment line 44 below and comment 45, to see two AIs using
 *         the same server.
 * 
 */
public class RAIGridnetClient {

    // Settings
    public RewardFunctionInterface[] rfs;
    String micrortsPath;
    public String mapPath;
    public AI ai2;
    UnitTypeTable utt;
    public boolean partialObs = false;

    // Internal State
    public PhysicalGameState pgs;
    public GameState gs;
    public GameState player1gs, player2gs;
    boolean gameover = false;
    boolean layerJSON = true;
    public int renderTheme = PhysicalGameStatePanel.COLORSCHEME_WHITE;
    public int maxAttackRadius;
    PhysicalGameStateJFrame w;
    public JNIInterface ai1;

    // storage
    int[][][] masks;
    double[] rewards;
    boolean[] dones;
    RAIResponse response;
    PlayerAction pa1;
    PlayerAction pa2;

    public RAIGridnetClient(RewardFunctionInterface[] a_rfs, String a_micrortsPath, String a_mapPath, AI a_ai2,
            UnitTypeTable a_utt, boolean partial_obs) throws Exception {
        micrortsPath = a_micrortsPath;
        mapPath = a_mapPath;
        rfs = a_rfs;
        utt = a_utt;
        partialObs = partial_obs;
        maxAttackRadius = utt.getMaxAttackRange() * 2 + 1;
        ai1 = new JNIAI(100, 0, utt);
        ai2 = a_ai2;
        if (ai2 == null) {
            throw new Exception("no ai2 was chosen");
        }
        if (micrortsPath.length() != 0) {
            this.mapPath = Paths.get(micrortsPath, mapPath).toString();
        }

        pgs = PhysicalGameState.load(mapPath, utt);

        // initialize storage
        masks = new int[pgs.getHeight()][pgs.getWidth()][1 + 6 + 4 + 4 + 4 + 4 + utt.getUnitTypes().size()
                + maxAttackRadius * maxAttackRadius];
        rewards = new double[rfs.length];
        dones = new boolean[rfs.length];
        response = new RAIResponse(null, null, null, null, null, null, null);
    }

    public byte[] render(boolean returnPixels) throws Exception {
        if (w == null) {
            w = PhysicalGameStatePanel.newVisualizer(gs, 640, 640, partialObs, renderTheme);
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

    public RAIResponse gameStep(int[][] action, int player) throws Exception {
        if (partialObs) {
            player1gs = new PartiallyObservableGameState(gs, player);
            player2gs = new PartiallyObservableGameState(gs, 1 - player);
        } else {
            player1gs = gs;
            player2gs = gs;
        }
        pa1 = ai1.getAction(player, player1gs, action);
        assert pa1.getActions().size() == action.length;
        try {
            pa2 = ai2.getAction(1 - player, player2gs);
        } catch (Exception e) {
            e.printStackTrace();
            pa2 = new PlayerAction();
            pa2.fillWithNones(player2gs, 1 - player, 1);
        }
        TraceEntry te = new TraceEntry(gs.getPhysicalGameState().clone(), gs.getTime());
        gs.issueSafe(pa1);
        te.addPlayerAction(pa1.clone());
        var errored = false;
        try {
            gs.issueSafe(pa2);
            te.addPlayerAction(pa2.clone());
        } catch (Throwable e) {
            System.err.println("Player 2 Error: " + e.getMessage());
            e.printStackTrace();
            // Remove all AI units, which should cause it to auto-lose.
            var player2units = gs.getUnits().stream().filter(u -> u.getPlayer() == 1 - player).toArray(Unit[]::new);
            for (Unit p2Unit : player2units) {
                gs.removeUnit(p2Unit);
            }
            errored = true;
        }

        if (!errored) {
            // simulate:
            gameover = gs.cycle();
        }
        if (gameover || errored) {
            // ai1.gameOver(gs.winner());
            ai2.gameOver(gs.winner());
            gameover = true;
        }

        for (int i = 0; i < rewards.length; i++) {
            rfs[i].computeReward(player, 1 - player, te, gs);
            dones[i] = rfs[i].isDone();
            rewards[i] = rfs[i].getReward();
        }

        var gsw = new GameStateWrapper(gs);
        response.set(
                gsw.getArrayObservation(player),
                gsw.getBinaryMask(player),
                rewards,
                dones,
                ai1.computeInfo(player, player2gs),
                null,
                gsw.getPlayerResources(player));
        return response;
    }

    public String sendUTT() throws Exception {
        Writer w = new StringWriter();
        utt.toJSON(w);
        return w.toString(); // now it works fine
    }

    public RAIResponse reset(int player) throws Exception {
        ai1.reset();
        ai2 = ai2.clone();
        ai2.reset();
        pgs = PhysicalGameState.load(mapPath, utt);
        gs = new GameState(pgs, utt);
        if (partialObs) {
            player1gs = new PartiallyObservableGameState(gs, player);
        } else {
            player1gs = gs;
        }

        for (int i = 0; i < rewards.length; i++) {
            rewards[i] = 0;
            dones[i] = false;
        }

        var gsw = new GameStateWrapper(gs);
        response.set(
                gsw.getArrayObservation(player),
                gsw.getBinaryMask(player),
                rewards,
                dones,
                "{}",
                gsw.getTerrain(),
                gsw.getPlayerResources(player));
        return response;
    }

    public void close() throws Exception {
        if (w != null) {
            w.dispose();
        }
    }
}
