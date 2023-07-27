package ai.rai;

public class RAIBotResponses extends RAIResponses {

    public int[][][] action;

    public RAIBotResponses(byte[][] observation, byte[][] mask, double reward[][], boolean done[][], byte terrain[][],
            byte resources[][], int[][][] action) {
        super(observation, mask, reward, done, terrain, resources);
        this.action = action;
    }

    public void set(byte[][] observation, byte[][] mask, double reward[][], boolean done[][], byte terrain[][],
            byte resources[][], int[][][] action) {
        super.set(observation, mask, reward, done, terrain, resources);
        this.action = action;
    }
}
