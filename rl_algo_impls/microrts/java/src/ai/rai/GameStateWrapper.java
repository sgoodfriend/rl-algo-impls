package ai.rai;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.ResourceUsage;
import rts.UnitAction;
import rts.UnitActionAssignment;
import rts.units.Unit;
import rts.units.UnitType;
import rts.units.UnitTypeTable;

public class GameStateWrapper {
    GameState gs;
    int debugLevel;

    int[][][][] vectorObservation;
    public static final int numVectorObservationFeatureMaps = 6;
    public static final int numArrayObservationFeatureMaps = 2 + numVectorObservationFeatureMaps - 1;
    int[][][][] masks;

    public GameStateWrapper(GameState a_gs) {
        this(a_gs, 0);
    }

    public GameStateWrapper(GameState a_gs, int a_debugLevel) {
        gs = a_gs;
        debugLevel = a_debugLevel;
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
                getValidActionArray(u, utt, masks[player][u.getY()][u.getX()], maxAttackDiameter, 1);
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

            getValidActionArray(u, utt, unitMask, maxAttackDiameter, 0);
            if (debugLevel > 0) {
                int baseUnitMask[] = new int[maskSize];
                UnitAction.getValidActionArray(u, gs, utt, baseUnitMask, maxAttackDiameter, 0);
                for (int j = 0; j < unitMask.length; ++j) {
                    if (unitMask[j] != 0 && baseUnitMask[j] == 0) {
                        System.err.println(
                                "Action mask for unit " + u + " is true for index " + j + " despite base mask not.");
                    }
                }
            }
            for (int j = 0; j < unitMask.length; ++j) {
                byteMask[idx + 2 + j] = (byte) unitMask[j];
            }

            Arrays.fill(unitMask, 0);
        }

        return byteMask;
    }

    public void getValidActionArray(Unit u, UnitTypeTable utt, int[] mask, int maxAttackRange,
            int idxOffset) {
        final List<UnitAction> uas = getUnitActions(u);
        int centerCoordinate = maxAttackRange / 2;
        int numUnitTypes = utt.getUnitTypes().size();
        for (UnitAction ua : uas) {
            mask[idxOffset + ua.getType()] = 1;
            switch (ua.getType()) {
                case UnitAction.TYPE_NONE: {
                    break;
                }
                case UnitAction.TYPE_MOVE: {
                    mask[idxOffset + UnitAction.NUMBER_OF_ACTION_TYPES + ua.getDirection()] = 1;
                    break;
                }
                case UnitAction.TYPE_HARVEST: {
                    // +4 offset --> slots for movement directions
                    mask[idxOffset + UnitAction.NUMBER_OF_ACTION_TYPES + 4 + ua.getDirection()] = 1;
                    break;
                }
                case UnitAction.TYPE_RETURN: {
                    // +4+4 offset --> slots for movement and harvest directions
                    mask[idxOffset + UnitAction.NUMBER_OF_ACTION_TYPES + 4 + 4 + ua.getDirection()] = 1;
                    break;
                }
                case UnitAction.TYPE_PRODUCE: {
                    // +4+4+4 offset --> slots for movement, harvest, and resource-return directions
                    mask[idxOffset + UnitAction.NUMBER_OF_ACTION_TYPES + 4 + 4 + 4 + ua.getDirection()] = 1;
                    // +4+4+4+4 offset --> slots for movement, harvest, resource-return, and
                    // unit-produce directions
                    mask[idxOffset + UnitAction.NUMBER_OF_ACTION_TYPES + 4 + 4 + 4 + 4 + ua.getUnitType().ID] = 1;
                    break;
                }
                case UnitAction.TYPE_ATTACK_LOCATION: {
                    int relative_x = ua.getLocationX() - u.getX();
                    int relative_y = ua.getLocationY() - u.getY();
                    // +4+4+4+4 offset --> slots for movement, harvest, resource-return, and
                    // unit-produce directions
                    mask[idxOffset + UnitAction.NUMBER_OF_ACTION_TYPES + 4 + 4 + 4 + 4 + numUnitTypes
                            + (centerCoordinate + relative_y) * maxAttackRange + (centerCoordinate + relative_x)] = 1;
                    break;
                }
            }
        }
    }

    public List<UnitAction> getUnitActions(Unit unit) {
        List<UnitAction> l = new ArrayList<>();

        PhysicalGameState pgs = gs.getPhysicalGameState();
        int player = unit.getPlayer();
        Player p = pgs.getPlayer(player);

        int x = unit.getX();
        int y = unit.getY();
        // retrieves units around me
        Unit uup = null, uright = null, udown = null, uleft = null;
        UnitActionAssignment uaaUp = null, uaaRight = null, uaaDown = null, uaaLeft = null;
        for (Unit u : pgs.getUnits()) {
            if (u.getX() == x) {
                if (u.getY() == y - 1) {
                    uup = u;
                    uaaUp = gs.getActionAssignment(uup);
                } else if (u.getY() == y + 1) {
                    udown = u;
                    uaaDown = gs.getActionAssignment(udown);
                }
            } else {
                if (u.getY() == y) {
                    if (u.getX() == x - 1) {
                        uleft = u;
                        uaaLeft = gs.getActionAssignment(uleft);
                    } else if (u.getX() == x + 1) {
                        uright = u;
                        uaaRight = gs.getActionAssignment(uright);
                    }
                }
            }
        }

        UnitType type = unit.getType();
        // if this unit can attack, adds an attack action for each unit around it
        if (type.canAttack) {
            int attackTime = type.attackTime;
            if (type.attackRange == 1) {
                if (y > 0 && uup != null && uup.getPlayer() != player && uup.getPlayer() >= 0
                        && !isUnitMovingWithinTimesteps(uaaUp, attackTime)) {
                    l.add(new UnitAction(UnitAction.TYPE_ATTACK_LOCATION, uup.getX(), uup.getY()));
                }
                if (x < pgs.getWidth() - 1 && uright != null && uright.getPlayer() != player
                        && uright.getPlayer() >= 0 && !isUnitMovingWithinTimesteps(uaaRight, attackTime)) {
                    l.add(new UnitAction(UnitAction.TYPE_ATTACK_LOCATION, uright.getX(), uright.getY()));
                }
                if (y < pgs.getHeight() - 1 && udown != null && udown.getPlayer() != player && udown.getPlayer() >= 0
                        && !isUnitMovingWithinTimesteps(uaaDown, attackTime)) {
                    l.add(new UnitAction(UnitAction.TYPE_ATTACK_LOCATION, udown.getX(), udown.getY()));
                }
                if (x > 0 && uleft != null && uleft.getPlayer() != player && uleft.getPlayer() >= 0
                        && !isUnitMovingWithinTimesteps(uaaLeft, attackTime)) {
                    l.add(new UnitAction(UnitAction.TYPE_ATTACK_LOCATION, uleft.getX(), uleft.getY()));
                }
            } else {
                int sqrange = type.attackRange * type.attackRange;
                for (Unit u : pgs.getUnits()) {
                    if (u.getPlayer() < 0 || u.getPlayer() == player) {
                        continue;
                    }
                    int sq_dx = (u.getX() - x) * (u.getX() - x);
                    int sq_dy = (u.getY() - y) * (u.getY() - y);
                    if (sq_dx + sq_dy <= sqrange) {
                        UnitActionAssignment uaa = gs.getActionAssignment(u);
                        if (!isUnitMovingWithinTimesteps(uaa, attackTime))
                            l.add(new UnitAction(UnitAction.TYPE_ATTACK_LOCATION, u.getX(), u.getY()));
                    }
                }
            }
        }

        int resources = unit.getResources();
        // if this unit can harvest, adds a harvest action for each resource around it
        // if it is already carrying resources, adds a return action for each allied
        // base around it
        if (type.canHarvest) {
            // harvest:
            if (resources == 0) {
                if (y > 0 && uup != null && uup.getType().isResource) {
                    l.add(new UnitAction(UnitAction.TYPE_HARVEST, UnitAction.DIRECTION_UP));
                }
                if (x < pgs.getWidth() - 1 && uright != null && uright.getType().isResource) {
                    l.add(new UnitAction(UnitAction.TYPE_HARVEST, UnitAction.DIRECTION_RIGHT));
                }
                if (y < pgs.getHeight() - 1 && udown != null && udown.getType().isResource) {
                    l.add(new UnitAction(UnitAction.TYPE_HARVEST, UnitAction.DIRECTION_DOWN));
                }
                if (x > 0 && uleft != null && uleft.getType().isResource) {
                    l.add(new UnitAction(UnitAction.TYPE_HARVEST, UnitAction.DIRECTION_LEFT));
                }
            }
            // return:
            if (resources > 0) {
                if (y > 0 && uup != null && uup.getType().isStockpile && uup.getPlayer() == player) {
                    l.add(new UnitAction(UnitAction.TYPE_RETURN, UnitAction.DIRECTION_UP));
                }
                if (x < pgs.getWidth() - 1 && uright != null && uright.getType().isStockpile
                        && uright.getPlayer() == player) {
                    l.add(new UnitAction(UnitAction.TYPE_RETURN, UnitAction.DIRECTION_RIGHT));
                }
                if (y < pgs.getHeight() - 1 && udown != null && udown.getType().isStockpile
                        && udown.getPlayer() == player) {
                    l.add(new UnitAction(UnitAction.TYPE_RETURN, UnitAction.DIRECTION_DOWN));
                }
                if (x > 0 && uleft != null && uleft.getType().isStockpile && uleft.getPlayer() == player) {
                    l.add(new UnitAction(UnitAction.TYPE_RETURN, UnitAction.DIRECTION_LEFT));
                }
            }
        }

        ResourceUsage base_ru = new ResourceUsage();
        for (Unit u : pgs.getUnits()) {
            UnitActionAssignment uaa = gs.getActionAssignment(u);
            if (uaa != null) {
                ResourceUsage ru = uaa.action.resourceUsage(u, pgs);
                base_ru.merge(ru);
            }
        }

        // if the player has enough resources, adds a produce action for each type this
        // unit produces.
        // a produce action is added for each free tile around the producer
        for (UnitType ut : type.produces) {
            if (p.getResources() >= ut.cost + base_ru.getResourcesUsed(player)) {
                int tup = (y > 0 ? pgs.getTerrain(x, y - 1) : PhysicalGameState.TERRAIN_WALL);
                int tright = (x < pgs.getWidth() - 1 ? pgs.getTerrain(x + 1, y) : PhysicalGameState.TERRAIN_WALL);
                int tdown = (y < pgs.getHeight() - 1 ? pgs.getTerrain(x, y + 1) : PhysicalGameState.TERRAIN_WALL);
                int tleft = (x > 0 ? pgs.getTerrain(x - 1, y) : PhysicalGameState.TERRAIN_WALL);

                if (tup == PhysicalGameState.TERRAIN_NONE && pgs.getUnitAt(x, y - 1) == null) {
                    var ua = new UnitAction(UnitAction.TYPE_PRODUCE, UnitAction.DIRECTION_UP, ut);
                    if (ua.resourceUsage(unit, pgs).consistentWith(base_ru, gs))
                        l.add(ua);
                }
                if (tright == PhysicalGameState.TERRAIN_NONE && pgs.getUnitAt(x + 1, y) == null) {
                    var ua = new UnitAction(UnitAction.TYPE_PRODUCE, UnitAction.DIRECTION_RIGHT, ut);
                    if (ua.resourceUsage(unit, pgs).consistentWith(base_ru, gs))
                        l.add(ua);
                }
                if (tdown == PhysicalGameState.TERRAIN_NONE && pgs.getUnitAt(x, y + 1) == null) {
                    var ua = new UnitAction(UnitAction.TYPE_PRODUCE, UnitAction.DIRECTION_DOWN, ut);
                    if (ua.resourceUsage(unit, pgs).consistentWith(base_ru, gs))
                        l.add(ua);
                }
                if (tleft == PhysicalGameState.TERRAIN_NONE && pgs.getUnitAt(x - 1, y) == null) {
                    var ua = new UnitAction(UnitAction.TYPE_PRODUCE, UnitAction.DIRECTION_LEFT, ut);
                    if (ua.resourceUsage(unit, pgs).consistentWith(base_ru, gs))
                        l.add(ua);
                }
            }
        }

        // if the unit can move, adds a move action for each free tile around it
        if (type.canMove) {
            int tup = (y > 0 ? pgs.getTerrain(x, y - 1) : PhysicalGameState.TERRAIN_WALL);
            int tright = (x < pgs.getWidth() - 1 ? pgs.getTerrain(x + 1, y) : PhysicalGameState.TERRAIN_WALL);
            int tdown = (y < pgs.getHeight() - 1 ? pgs.getTerrain(x, y + 1) : PhysicalGameState.TERRAIN_WALL);
            int tleft = (x > 0 ? pgs.getTerrain(x - 1, y) : PhysicalGameState.TERRAIN_WALL);

            if (tup == PhysicalGameState.TERRAIN_NONE && uup == null) {
                var ua = new UnitAction(UnitAction.TYPE_MOVE, UnitAction.DIRECTION_UP);
                if (ua.resourceUsage(unit, pgs).consistentWith(base_ru, gs))
                    l.add(ua);
            }
            if (tright == PhysicalGameState.TERRAIN_NONE && uright == null) {
                var ua = new UnitAction(UnitAction.TYPE_MOVE, UnitAction.DIRECTION_RIGHT);
                if (ua.resourceUsage(unit, pgs).consistentWith(base_ru, gs))
                    l.add(ua);
            }
            if (tdown == PhysicalGameState.TERRAIN_NONE && udown == null) {
                var ua = new UnitAction(UnitAction.TYPE_MOVE, UnitAction.DIRECTION_DOWN);
                if (ua.resourceUsage(unit, pgs).consistentWith(base_ru, gs))
                    l.add(ua);
            }
            if (tleft == PhysicalGameState.TERRAIN_NONE && uleft == null) {
                var ua = new UnitAction(UnitAction.TYPE_MOVE, UnitAction.DIRECTION_LEFT);
                if (ua.resourceUsage(unit, pgs).consistentWith(base_ru, gs))
                    l.add(ua);
            }
        }

        // units can always stay idle:
        l.add(new UnitAction(UnitAction.TYPE_NONE, 1));

        return l;
    }

    public boolean isUnitMovingWithinTimesteps(UnitActionAssignment uaa, int timesteps) {
        if (uaa == null || uaa.action.getType() != UnitAction.TYPE_MOVE) {
            return false;
        }
        int timestepsToCompletion = uaa.time + uaa.action.ETA(uaa.unit) - gs.getTime();
        return timestepsToCompletion < timesteps;
    }
}
