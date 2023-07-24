package ai.rai;

public class RAIBotResponse extends RAIResponse {
    public int[][] action;

    public RAIBotResponse(byte[] observation, byte[] mask, double reward[], boolean done[], String info, byte[] terrain,
            byte[] resources, int[][] action) {
        super(observation, mask, reward, done, info, terrain, resources);
        this.action = action;
    }

    public void set(byte[] observation, byte[] mask, double reward[], boolean done[], String info, byte[] terrain,
            byte[] resources, int[][] action) {
        super.set(observation, mask, reward, done, info, terrain, resources);
        this.action = action;
    }
}
