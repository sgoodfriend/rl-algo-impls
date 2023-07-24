/*
* To change this template, choose Tools | Templates
* and open the template in the editor.
*/
package tests;

import java.io.Writer;
import java.nio.file.Paths;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.awt.image.BufferedImage;
import java.io.StringWriter;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;

import ai.PassiveAI;
import ai.RandomBiasedAI;
import ai.core.AI;
import ai.jni.JNIAI;
import ai.reward.RewardFunctionInterface;
import ai.jni.JNIInterface;
import ai.rai.GameStateWrapper;
import ai.rai.RAIResponse;
import ai.rai.RAIResponses;
import gui.PhysicalGameStateJFrame;
import gui.PhysicalGameStatePanel;
import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.Trace;
import rts.TraceEntry;
import rts.UnitAction;
import rts.UnitActionAssignment;
import rts.units.Unit;
import rts.units.UnitTypeTable;
import tests.RAIGridnetClientSelfPlay;

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
public class RAIGridnetVecClient {
    public RAIGridnetClient[] clients;
    public RAIGridnetClientSelfPlay[] selfPlayClients;
    public int maxSteps;
    public int[] envSteps;
    public RewardFunctionInterface[] rfs;
    public UnitTypeTable utt;
    boolean partialObs = false;
    public String[] mapPaths;

    // storage
    byte[][] mask;
    byte[][] observation;
    double[][] reward;
    boolean[][] done;
    byte[][] resources;
    RAIResponse[] rs;
    RAIResponses responses;
    byte[][] terrain;

    double[] terminalReward1;
    boolean[] terminalDone1;
    double[] terminalReward2;
    boolean[] terminalDone2;

    public RAIGridnetVecClient(int a_num_selfplayenvs, int a_num_envs, int a_max_steps, RewardFunctionInterface[] a_rfs,
            String a_micrortsPath, String[] a_mapPaths,
            AI[] a_ai2s, UnitTypeTable a_utt, boolean partial_obs) throws Exception {
        maxSteps = a_max_steps;
        utt = a_utt;
        rfs = a_rfs;
        partialObs = partial_obs;
        mapPaths = a_mapPaths;

        // initialize clients
        envSteps = new int[a_num_selfplayenvs + a_num_envs];
        selfPlayClients = new RAIGridnetClientSelfPlay[a_num_selfplayenvs / 2];
        for (int i = 0; i < selfPlayClients.length; i++) {
            selfPlayClients[i] = new RAIGridnetClientSelfPlay(a_rfs, a_micrortsPath, mapPaths[i * 2], a_utt,
                    partialObs);
        }
        clients = new RAIGridnetClient[a_num_envs];
        for (int i = 0; i < clients.length; i++) {
            clients[i] = new RAIGridnetClient(a_rfs, a_micrortsPath, mapPaths[a_num_selfplayenvs + i], a_ai2s[i], a_utt,
                    partialObs);
        }

        int s1 = a_num_selfplayenvs + a_num_envs;
        mask = new byte[s1][];
        observation = new byte[s1][];
        reward = new double[s1][rfs.length];
        done = new boolean[s1][rfs.length];
        resources = new byte[s1][2];
        terminalReward1 = new double[rfs.length];
        terminalDone1 = new boolean[rfs.length];
        terminalReward2 = new double[rfs.length];
        terminalDone2 = new boolean[rfs.length];
        responses = new RAIResponses(null, null, null, null, null, null);
        terrain = new byte[s1][];
        rs = new RAIResponse[s1];
    }

    public RAIResponses reset(int[] players) throws Exception {
        for (int i = 0; i < selfPlayClients.length; i++) {
            selfPlayClients[i].reset();
            for (int p = 0; p < 2; ++p) {
                rs[i * 2 + p] = selfPlayClients[i].getResponse(p);
            }
        }
        Arrays.fill(envSteps, 0);
        for (int i = selfPlayClients.length * 2; i < players.length; i++) {
            rs[i] = clients[i - selfPlayClients.length * 2].reset(players[i]);
        }

        for (int i = 0; i < rs.length; i++) {
            observation[i] = rs[i].observation;
            mask[i] = rs[i].mask;
            reward[i] = rs[i].reward;
            done[i] = rs[i].done;
            terrain[i] = rs[i].terrain;
            resources[i] = rs[i].resources;
        }
        responses.set(observation, mask, reward, done, terrain, resources);
        return responses;
    }

    public RAIResponses gameStep(int[][][] action, int[] players) throws Exception {
        for (int i = 0; i < selfPlayClients.length; i++) {
            selfPlayClients[i].gameStep(action[i * 2], action[i * 2 + 1]);
            rs[i * 2] = selfPlayClients[i].getResponse(0);
            rs[i * 2 + 1] = selfPlayClients[i].getResponse(1);
            envSteps[i * 2] += 1;
            envSteps[i * 2 + 1] += 1;
            if (rs[i * 2].done[0] || envSteps[i * 2] >= maxSteps) {
                for (int j = 0; j < terminalReward1.length; j++) {
                    terminalReward1[j] = rs[i * 2].reward[j];
                    terminalDone1[j] = rs[i * 2].done[j];
                    terminalReward2[j] = rs[i * 2 + 1].reward[j];
                    terminalDone2[j] = rs[i * 2 + 1].done[j];
                }

                selfPlayClients[i].reset();
                for (int j = 0; j < terminalReward1.length; j++) {
                    rs[i * 2].reward[j] = terminalReward1[j];
                    rs[i * 2].done[j] = terminalDone1[j];
                    rs[i * 2 + 1].reward[j] = terminalReward2[j];
                    rs[i * 2 + 1].done[j] = terminalDone2[j];
                }
                rs[i * 2].done[0] = true;
                rs[i * 2 + 1].done[0] = true;
                envSteps[i * 2] = 0;
                envSteps[i * 2 + 1] = 0;
            }
        }

        for (int i = selfPlayClients.length * 2; i < players.length; i++) {
            envSteps[i] += 1;
            rs[i] = clients[i - selfPlayClients.length * 2].gameStep(action[i], players[i]);
            if (rs[i].done[0] || envSteps[i] >= maxSteps) {
                // TRICKY: note that `clients` already resets the shared `observation`
                // so we need to set the old reward and done to this response
                for (int j = 0; j < rs[i].reward.length; j++) {
                    terminalReward1[j] = rs[i].reward[j];
                    terminalDone1[j] = rs[i].done[j];
                }
                clients[i - selfPlayClients.length * 2].reset(players[i]);
                for (int j = 0; j < rs[i].reward.length; j++) {
                    rs[i].reward[j] = terminalReward1[j];
                    rs[i].done[j] = terminalDone1[j];
                }
                rs[i].done[0] = true;
                envSteps[i] = 0;
            }
        }
        for (int i = 0; i < rs.length; i++) {
            observation[i] = rs[i].observation;
            mask[i] = rs[i].mask;
            reward[i] = rs[i].reward;
            done[i] = rs[i].done;
            resources[i] = rs[i].resources;
        }
        responses.set(observation, mask, reward, done, null, resources);
        return responses;
    }

    public void close() throws Exception {
        if (clients != null) {
            for (RAIGridnetClient client : clients) {
                client.close();
            }
        }
        if (selfPlayClients != null) {
            for (RAIGridnetClientSelfPlay client : selfPlayClients) {
                client.close();
            }
        }
    }
}
