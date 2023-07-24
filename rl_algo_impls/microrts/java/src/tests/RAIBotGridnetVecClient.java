package tests;

import java.util.Arrays;

import ai.core.AI;
import ai.rai.RAIBotResponse;
import ai.rai.RAIBotResponses;
import ai.reward.RewardFunctionInterface;
import rts.units.UnitTypeTable;

public class RAIBotGridnetVecClient {
    public RAIBotClient[] botClients;

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
    RAIBotResponse[] rs;
    RAIBotResponses responses;
    byte[][] terrain;
    int[][][] action;

    double[] terminalReward1;
    boolean[] terminalDone1;
    int[][] terminalAction1;
    double[] terminalReward2;
    boolean[] terminalDone2;
    int[][] terminalAction2;

    /**
     * Constructor for Java-bot-only environments.
     * 
     * @param a_max_steps
     * @param a_rfs
     * @param a_micrortsPath
     * @param a_mapPaths
     * @param a_ais
     * @param a_utt
     * @param partial_obs
     * @throws Exception
     */
    public RAIBotGridnetVecClient(int a_max_steps, RewardFunctionInterface[] a_rfs, String a_micrortsPath,
            String[] a_mapPaths,
            AI[] a_ais, UnitTypeTable a_utt, boolean partial_obs) throws Exception {
        maxSteps = a_max_steps;
        utt = a_utt;
        rfs = a_rfs;
        partialObs = partial_obs;
        mapPaths = a_mapPaths;

        // initialize clients
        botClients = new RAIBotClient[a_ais.length / 2];
        for (int i = 0; i < botClients.length; i++) {
            botClients[i] = new RAIBotClient(a_rfs, a_micrortsPath, mapPaths[2 * i], a_ais[2 * i], a_ais[2 * i + 1],
                    a_utt,
                    partialObs);
        }

        int s1 = botClients.length * 2;
        envSteps = new int[s1];
        mask = new byte[s1][];
        observation = new byte[s1][];
        reward = new double[s1][rfs.length];
        done = new boolean[s1][rfs.length];
        resources = new byte[s1][2];
        terminalReward1 = new double[rfs.length];
        terminalDone1 = new boolean[rfs.length];
        terminalReward2 = new double[rfs.length];
        terminalDone2 = new boolean[rfs.length];
        responses = new RAIBotResponses(null, null, null, null, null, null, null);
        terrain = new byte[s1][];
        action = new int[s1][][];
        rs = new RAIBotResponse[s1];
    }

    public RAIBotResponses reset() throws Exception {
        for (int i = 0; i < botClients.length; i++) {
            botClients[i].reset();
            for (int p = 0; p < 2; ++p) {
                rs[i * 2 + p] = botClients[i].getResponse(p);
            }
        }
        Arrays.fill(envSteps, 0);

        for (int i = 0; i < rs.length; i++) {
            observation[i] = rs[i].observation;
            mask[i] = rs[i].mask;
            reward[i] = rs[i].reward;
            done[i] = rs[i].done;
            terrain[i] = rs[i].terrain;
            resources[i] = rs[i].resources;
        }
        responses.set(observation, mask, reward, done, terrain, resources, null);
        return responses;
    }

    public RAIBotResponses gameStep() throws Exception {
        for (int i = 0; i < botClients.length; i++) {
            botClients[i].gameStep();
            for (int p = 0; p < 2; ++p) {
                rs[i * 2 + p] = botClients[i].getResponse(p);
                envSteps[i * 2 + p] += 1;
            }
            if (rs[i * 2].done[0] || envSteps[i * 2] >= maxSteps) {
                for (int j = 0; j < terminalReward1.length; j++) {
                    terminalReward1[j] = rs[i * 2].reward[j];
                    terminalDone1[j] = rs[i * 2].done[j];
                    terminalReward2[j] = rs[i * 2 + 1].reward[j];
                    terminalDone2[j] = rs[i * 2 + 1].done[j];
                }
                terminalAction1 = rs[i * 2].action;
                terminalAction2 = rs[i * 2 + 1].action;

                botClients[i].reset();
                for (int j = 0; j < terminalReward1.length; j++) {
                    rs[i * 2].reward[j] = terminalReward1[j];
                    rs[i * 2].done[j] = terminalDone1[j];
                    rs[i * 2 + 1].reward[j] = terminalReward2[j];
                    rs[i * 2 + 1].done[j] = terminalDone2[j];
                }
                rs[i * 2].done[0] = true;
                rs[i * 2 + 1].done[0] = true;
                rs[i * 2].action = terminalAction1;
                rs[i * 2 + 1].action = terminalAction2;
                envSteps[i * 2] = 0;
                envSteps[i * 2 + 1] = 0;
            }
        }
        for (int i = 0; i < rs.length; i++) {
            observation[i] = rs[i].observation;
            mask[i] = rs[i].mask;
            reward[i] = rs[i].reward;
            done[i] = rs[i].done;
            resources[i] = rs[i].resources;
            action[i] = rs[i].action;
        }
        responses.set(observation, mask, reward, done, null, resources, action);
        return responses;
    }

    public void close() throws Exception {
        for (int i = 0; i < botClients.length; i++) {
            botClients[i].close();
        }
    }
}
