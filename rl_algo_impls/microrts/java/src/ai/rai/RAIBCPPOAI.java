package ai.rai;

import ai.core.AI;
import rts.units.UnitTypeTable;

public class RAIBCPPOAI extends RAISocketAI {
    public RAIBCPPOAI(UnitTypeTable a_utt) {
        super(100, -1, a_utt, 0, 1, false, "RAI-BC-PPO");
    }

    public RAIBCPPOAI(int mt, int mi, UnitTypeTable a_utt) {
        super(mt, mi, a_utt, 0, 1, false, "RAI-BC-PPO");
    }

    public RAIBCPPOAI(int mt, int mi, UnitTypeTable a_utt, int overrideTorchThreads,
            int pythonVerboseLevel) {
        super(mt, mi, a_utt, overrideTorchThreads, pythonVerboseLevel, false, "RAI-BC-PPO");
    }

    public RAIBCPPOAI(int mt, int mi, UnitTypeTable a_utt, int overrideTorchThreads,
            int pythonVerboseLevel,
            boolean useBestModels) {
        super(mt, mi, a_utt, overrideTorchThreads, pythonVerboseLevel, useBestModels, "RAI-BC-PPO");
    }

    public AI clone() {
        if (DEBUG >= 1)
            System.out.println("RAIBCPPOAI: cloning");
        return new RAIBCPPOAI(TIME_BUDGET, ITERATIONS_BUDGET, utt, OVERRIDE_TORCH_THREADS,
                PYTHON_VERBOSE_LEVEL);
    }
}
