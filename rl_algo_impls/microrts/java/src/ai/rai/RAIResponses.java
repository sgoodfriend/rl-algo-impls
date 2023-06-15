package ai.rai;

public class RAIResponses {
    public byte[][] observation;
    public byte[][] mask;
    public double[][] reward;
    public boolean[][] done;
    public byte[][] terrain;

    public RAIResponses(byte[][] observation, byte[][] mask, double reward[][], boolean done[][], byte terrain[][]) {
        this.observation = observation;
        this.mask = mask;
        this.reward = reward;
        this.done = done;
        this.terrain = terrain;
    }

    public void set(byte[][] observation, byte[][] mask, double reward[][], boolean done[][], byte terrain[][]) {
        this.observation = observation;
        this.mask = mask;
        this.reward = reward;
        this.done = done;
        this.terrain = terrain;
    }

}
