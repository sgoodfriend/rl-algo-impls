package ai.reward;

import rts.GameState;
import rts.TraceEntry;
import rts.units.Unit;

/**
 * @author sgoodfriend
 */
public class ScoreRewardFunction extends RewardFunctionInterface {
    public void computeReward(int maxplayer, int minplayer, TraceEntry te, GameState afterGs) {
        reward = 0.0;
        done = afterGs.gameover();
        double ownScore = 0;
        double oppScore = 0;
        for (Unit u : afterGs.getUnits()) {
            double unitScore = u.getCost() * (1 + (double) u.getHitPoints() / u.getMaxHitPoints());
            if (u.getPlayer() == maxplayer) {
                ownScore += unitScore;
            } else if (u.getPlayer() == minplayer) {
                oppScore += unitScore;
            }
        }
        reward = (ownScore - oppScore) / (ownScore + oppScore + 1);
    }
}
