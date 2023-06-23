package tournaments;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.Writer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

import ai.RandomBiasedAI;
import ai.abstraction.partialobservability.POLightRush;
import ai.abstraction.partialobservability.POWorkerRush;
import ai.coac.CoacAI;
import ai.core.AI;
import ai.mcts.naivemcts.NaiveMCTS;
import ai.rai.RAISocketAI;
import mayariBot.mayari;
import rts.units.UnitTypeTable;

public class RAIRoundRobinTournament {
    public static void main(String args[]) throws Exception {
        final UnitTypeTable utt = new UnitTypeTable(UnitTypeTable.VERSION_ORIGINAL_FINETUNED);
        final AI[] AIs = {
                new RAISocketAI(utt),
                new RandomBiasedAI(utt),
                new NaiveMCTS(utt),
                new POWorkerRush(utt),
                new POLightRush(utt),
                new CoacAI(utt),
                new mayari(utt)
        };
        final String[] maps = {
                "maps/BWDistantResources32x32.xml"
        };
        final int maxGameLength = 6000;

        String prefix = "tournament_";
        if (maps.length == 1) {
            Path mapPath = Paths.get(maps[0]);
            String nameWithExtension = mapPath.getFileName().toString();
            String nameWithoutExtension = nameWithExtension.substring(0, nameWithExtension.lastIndexOf("."));
            prefix += nameWithoutExtension + "_";
        }
        int idx = 0;
        File file;
        do {
            idx++;
            file = new File(prefix + idx);
        } while (file.exists());
        file.mkdir();
        String tournamentfolder = file.getName();
        final File fileToUse = new File(tournamentfolder + "/tournament.csv");
        final String tracesFolder = tournamentfolder + "/traces";

        final boolean onlyPlayFirstAI = true;
        final int iterations = 10;
        final int timeBudget = 100;
        final int iterationsBudget = -1;
        final int preAnalysisBudgetFirstTimeInAMap = 1000;
        final int preAnalysisBudgetRestOfTimes = 1000;
        final boolean fullObservability = true;
        final boolean selfMatches = false;
        final boolean timeoutCheck = false;
        final boolean runGC = true;
        final boolean preAnalysis = preAnalysisBudgetFirstTimeInAMap > 0;
        final Writer out = new FileWriter(fileToUse);
        final Writer progress = new PrintWriter(System.out);
        RoundRobinTournament tournament = new RoundRobinTournament(Arrays.asList(AIs));
        tournament.runTournament(
                onlyPlayFirstAI ? 0 : -1,
                Arrays.asList(maps),
                iterations,
                maxGameLength,
                timeBudget,
                iterationsBudget,
                preAnalysisBudgetFirstTimeInAMap,
                preAnalysisBudgetRestOfTimes,
                fullObservability,
                selfMatches,
                timeoutCheck,
                runGC,
                preAnalysis,
                utt,
                tracesFolder,
                out,
                progress,
                tournamentfolder);
    }
}
