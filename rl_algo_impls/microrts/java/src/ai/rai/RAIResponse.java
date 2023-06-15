package ai.rai;

public class RAIResponse {
    public byte[] observation;
    public byte[] mask;
    public double[] reward;
    public boolean[] done;
    public String info;
    public byte[] terrain;

    public RAIResponse(byte[] observation, byte[] mask, double reward[], boolean done[], String info, byte[] terrain) {
        this.observation = observation;
        this.mask = mask;
        this.reward = reward;
        this.done = done;
        this.info = info;
        this.terrain = terrain;
    }

    public void set(byte[] observation, byte[] mask, double reward[], boolean done[], String info, byte[] terrain) {
        this.observation = observation;
        this.mask = mask;
        this.reward = reward;
        this.done = done;
        this.info = info;
        this.terrain = terrain;
    }
}
