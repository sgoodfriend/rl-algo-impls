package ai.rai;

public class RAIResponses {
    public byte[][] observation;
    public byte[][] mask;
    public double[][] reward;
    public boolean[][] done;
    public byte[][] terrain;
    public byte[][] resources;

    public RAIResponses(byte[][] observation, byte[][] mask, double reward[][], boolean done[][], byte terrain[][],
            byte resources[][]) {
        this.observation = observation;
        this.mask = mask;
        this.reward = reward;
        this.done = done;
        this.terrain = terrain;
        this.resources = resources;
    }

    public void set(byte[][] observation, byte[][] mask, double reward[][], boolean done[][], byte terrain[][],
            byte resources[][]) {
        this.observation = observation;
        this.mask = mask;
        this.reward = reward;
        this.done = done;
        this.terrain = terrain;
        this.resources = resources;
    }

}
