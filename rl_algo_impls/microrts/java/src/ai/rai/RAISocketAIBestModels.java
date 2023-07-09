package ai.rai;

import rts.units.UnitTypeTable;

public class RAISocketAIBestModels extends RAISocketAI {
    public RAISocketAIBestModels(UnitTypeTable a_utt) {
        super(100, -1, a_utt, 0, 1, true, false);
    }

    public RAISocketAIBestModels(int mt, int mi, UnitTypeTable a_utt) {
        super(mt, mi, a_utt, 0, 1, true, false);
    }
}
