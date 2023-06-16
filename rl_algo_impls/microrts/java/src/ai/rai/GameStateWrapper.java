package ai.rai;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

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
    public static final int numArrayObservationFeatureMaps = 2 + numVectorObservationFeatureMaps - 1;
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
            int unitPlayer = u.getPlayer();
            if (unitPlayer != -1) {
                vectorObservation[player][2][u.getY()][u.getX()] = (unitPlayer + player) % 2;
            }
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

    public byte[] getTerrain() {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int height = pgs.getHeight();
        int width = pgs.getWidth();

        byte walls[] = new byte[height * width];
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                walls[y * width + x] = (byte) pgs.getTerrain(x, y);
            }
        }
        return walls;
    }

    /**
     * Constructs an array observation for a player (length number of units)
     * 
     * | Observation Features | Max | Values
     * |----------------------|------------------------------------------------------------------|
     * | y | 128 | Java's bytes are signed, while Python's are unsigned
     * | x | 128 |
     * | Hit Points | 10
     * | Resources | 40
     * | Owner | 3 | -, player 1, player 2)
     * | Unit Types | 8 | -, resource, base, barrack, worker, light, heavy, ranged
     * | Current Action | 6 | -, move, harvest, return, produce, attack
     * 
     * @param player
     * @return a vector observation for the specified player
     */
    public byte[] getArrayObservation(int player) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        List<Unit> units = pgs.getUnits();
        byte arrayObs[] = new byte[units.size() * numArrayObservationFeatureMaps];
        for (int i = 0; i < units.size(); ++i) {
            Unit u = units.get(i);
            int idx = i * numArrayObservationFeatureMaps;
            arrayObs[idx + 0] = (byte) u.getY();
            arrayObs[idx + 1] = (byte) u.getX();
            arrayObs[idx + 2] = (byte) u.getHitPoints();
            arrayObs[idx + 3] = (byte) u.getResources();
            int unitPlayer = u.getPlayer();
            if (unitPlayer != -1) {
                arrayObs[idx + 4] = (byte) ((unitPlayer + player) % 2 + 1);
            }
            arrayObs[idx + 5] = (byte) (u.getType().ID + 1);
            UnitActionAssignment uaa = gs.getActionAssignment(u);
            if (uaa != null) {
                arrayObs[idx + 6] = (byte) uaa.action.getType();
            } else {
                arrayObs[idx + 6] = (byte) UnitAction.TYPE_NONE;
            }
        }
        return arrayObs;
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

    public byte[] getBinaryMask(int player) {
        UnitTypeTable utt = gs.getUnitTypeTable();
        PhysicalGameState pgs = gs.getPhysicalGameState();
        List<Unit> units = pgs.getUnits().stream()
                .filter(u -> u.getPlayer() == player && gs.getActionAssignment(u) == null).collect(Collectors.toList());

        int maxAttackDiameter = utt.getMaxAttackRange() * 2 + 1;
        final int maskSize = 6 + 4 + 4 + 4 + 4 + utt.getUnitTypes().size()
                + maxAttackDiameter * maxAttackDiameter;
        byte byteMask[] = new byte[units.size() * (2 + maskSize)];
        int unitMask[] = new int[maskSize];
        for (int i = 0; i < units.size(); ++i) {
            Unit u = units.get(i);
            int idx = i * (2 + maskSize);
            byteMask[idx + 0] = (byte) u.getY();
            byteMask[idx + 1] = (byte) u.getX();

            UnitAction.getValidActionArray(u, gs, utt, unitMask, maxAttackDiameter, 0);
            for (int j = 0; j < unitMask.length; ++j) {
                byteMask[idx + 2 + j] = (byte) unitMask[j];
            }

            Arrays.fill(unitMask, 0);
        }

        return byteMask;
    }
}
