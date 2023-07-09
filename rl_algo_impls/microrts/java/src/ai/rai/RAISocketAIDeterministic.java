package ai.rai;

import rts.units.UnitTypeTable;

public class RAISocketAIDeterministic extends RAISocketAI {
    public RAISocketAIDeterministic(UnitTypeTable a_utt) {
        super(100, -1, a_utt, 0, 1, false, true);
    }

    public RAISocketAIDeterministic(int mt, int mi, UnitTypeTable a_utt) {
        super(mt, mi, a_utt, 0, 1, false, true);
    }
}
