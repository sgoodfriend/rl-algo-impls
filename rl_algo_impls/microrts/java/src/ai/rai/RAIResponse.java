package ai.rai;

public class RAIResponse {
    public byte[] observation;
    public byte[] mask;
    public double[] reward;
    public boolean[] done;
    public String info;
    public byte[] terrain;
    public byte[] resources;

    public RAIResponse(byte[] observation, byte[] mask, double reward[], boolean done[], String info, byte[] terrain,
            byte[] resources) {
        this.observation = observation;
        this.mask = mask;
        this.reward = reward;
        this.done = done;
        this.info = info;
        this.terrain = terrain;
        this.resources = resources;
    }

    public void set(byte[] observation, byte[] mask, double reward[], boolean done[], String info, byte[] terrain,
            byte[] resources) {
        this.observation = observation;
        this.mask = mask;
        this.reward = reward;
        this.done = done;
        this.info = info;
        this.terrain = terrain;
        this.resources = resources;
    }
}
