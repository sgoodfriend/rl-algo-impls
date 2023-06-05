package ai.rai;

import java.util.Arrays;
import java.util.List;

import rts.GameState;
import rts.PhysicalGameState;
import rts.UnitAction;
import rts.UnitActionAssignment;
import rts.units.Unit;
import rts.units.UnitTypeTable;

public class GameStateWrapper {
    GameState gs;

    int[][][][] vectorObservation;
    public static final int numVectorObservationFeatureMaps = 6;

    int[][][][] masks;

    public GameStateWrapper(GameState a_gs) {
        gs = a_gs;
    }

    /**
     * Constructs a vector observation for a player
     * 
     * | Observation Features | Max | Values
     * |----------------------|------------------------------------------------------------------|
     * | Hit Points | 10
     * | Resources | 40
     * | Owner | 3 | -, player 1, player 2)
     * | Unit Types | 8 | -, resource, base, barrack, worker, light, heavy, ranged
     * | Current Action | 6 | -, move, harvest, return, produce, attack
     * | Terrain | 2 | empty, wall
     * 
     * @param player
     * @return a vector observation for the specified player
     */
    public int[][][] getVectorObservation(int player) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int height = pgs.getHeight();
        int width = pgs.getWidth();
        if (vectorObservation == null) {
            vectorObservation = new int[2][numVectorObservationFeatureMaps][height][width];
        }
        // hitpointsMatrix is vectorObservation[player][0]
        // resourcesMatrix is vectorObservation[player][1]
        // playersMatrix is vectorObservation[player][2]
        // unitTypesMatrix is vectorObservation[player][3]
        // unitActionMatrix is vectorObservation[player][4]
        // terrainMatrix is matrixObservation[player][5]

        for (int i = 0; i < vectorObservation[player][0].length; i++) {
            Arrays.fill(vectorObservation[player][0][i], 0);
            Arrays.fill(vectorObservation[player][1][i], 0);
            Arrays.fill(vectorObservation[player][4][i], 0);
            Arrays.fill(vectorObservation[player][5][i], 0);
            // temp default value for empty spaces
            Arrays.fill(vectorObservation[player][2][i], -1);
            Arrays.fill(vectorObservation[player][3][i], -1);
        }

        List<Unit> units = pgs.getUnits();
        for (int i = 0; i < units.size(); i++) {
            Unit u = units.get(i);
            UnitActionAssignment uaa = gs.getActionAssignment(u);
            vectorObservation[player][0][u.getY()][u.getX()] = u.getHitPoints();
            vectorObservation[player][1][u.getY()][u.getX()] = u.getResources();
            vectorObservation[player][2][u.getY()][u.getX()] = (u.getPlayer() + player) % 2;
            vectorObservation[player][3][u.getY()][u.getX()] = u.getType().ID;
            if (uaa != null) {
                vectorObservation[player][4][u.getY()][u.getX()] = uaa.action.getType();
            } else {
                vectorObservation[player][4][u.getY()][u.getX()] = UnitAction.TYPE_NONE;
            }
        }

        // normalize by getting rid of -1
        for (int i = 0; i < vectorObservation[player][2].length; i++) {
            for (int j = 0; j < vectorObservation[player][2][i].length; j++) {
                vectorObservation[player][3][i][j] += 1;
                vectorObservation[player][2][i][j] += 1;
            }
        }

        // Terrain
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                vectorObservation[player][5][y][x] = pgs.getTerrain(x, y);
            }
        }

        return vectorObservation[player];
    }

    public int[][][] getMasks(int player) {
        UnitTypeTable utt = gs.getUnitTypeTable();
        PhysicalGameState pgs = gs.getPhysicalGameState();

        int maxAttackDiameter = utt.getMaxAttackRange() * 2 + 1;
        if (masks == null) {
            masks = new int[2][pgs.getHeight()][pgs.getWidth()][1 + 6 + 4 + 4 + 4 + 4 + utt.getUnitTypes().size()
                    + maxAttackDiameter * maxAttackDiameter];
        }

        Arrays.stream(masks[player]).forEach(mY -> Arrays.stream(mY).forEach(
                mX -> Arrays.fill(mX, 0)));

        for (Unit u : pgs.getUnits()) {
            final UnitActionAssignment uaa = gs.getActionAssignment(u);
            if (u.getPlayer() == player && uaa == null) {
                masks[player][u.getY()][u.getX()][0] = 1;
                UnitAction.getValidActionArray(u, gs, utt, masks[player][u.getY()][u.getX()], maxAttackDiameter, 1);
            }
        }

        return masks[player];
    }
}
