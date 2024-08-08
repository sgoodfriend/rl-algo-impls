package tournaments;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.Writer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import ai.abstraction.partialobservability.POLightRush;
import ai.abstraction.partialobservability.POWorkerRush;
import ai.coac.CoacAI;
import ai.core.AI;
import ai.rai.RAIBCPPOAI;
import ai.rai.RAISocketAI;
import mayariBot.mayari;
import rts.PhysicalGameState;
import rts.units.UnitTypeTable;
import util.Pair;

public class RAIPerformanceTournament extends RAITournament {
    public RAIPerformanceTournament(List<AI> AIs) {
        super(AIs);
    }

    public static void main(String args[]) throws Exception {
        RAITournamentArguments raiArgs = new RAITournamentArguments(args);
        int overrideTorchThreads = raiArgs.getOptionInteger('p', 0);
        int pythonVerboseLevel = raiArgs.pythonVerbosity;
        boolean useBestModels = raiArgs.hasOption('b');
        String modelSet = raiArgs.getOptionValue('m', "RAISocketAI");
        System.out.println("modelSet: " + modelSet);

        final int timeBudget = raiArgs.getOptionInteger('t', 100);
        final boolean timeoutCheck = !raiArgs.hasOption('T');
        final UnitTypeTable utt = new UnitTypeTable(UnitTypeTable.VERSION_ORIGINAL_FINETUNED);

        AI raiAI;
        if (modelSet.equals("RAI-BC-PPO")) {
            raiAI = new RAIBCPPOAI(timeBudget, -1, utt, overrideTorchThreads, pythonVerboseLevel, useBestModels);
        } else {
            raiAI = new RAISocketAI(timeBudget, -1, utt, overrideTorchThreads, pythonVerboseLevel, useBestModels,
                    modelSet);
        }
        final AI[] AIs = {
                raiAI,
                new CoacAI(utt),
                new mayari(utt),
        };
        var tournament = new RAIPerformanceTournament(Arrays.asList(AIs));

        final List<Pair<String, Integer>> maps = new ArrayList<Pair<String, Integer>>();
        maps.add(new Pair<>("maps/NoWhereToRun9x8.xml", 5000));
        maps.add(new Pair<>("maps/16x16/TwoBasesBarracks16x16.xml", 5000));
        maps.add(new Pair<>("maps/DoubleGame24x24.xml", 6000));
        maps.add(new Pair<>("maps/BWDistantResources32x32.xml", 7000));
        maps.add(new Pair<>("maps/BroodWar/(4)BloodBath.scmB.xml", 10000));
        maps.add(new Pair<>("maps/BroodWar/(4)Andromeda.scxB.xml", 14000));

        String prefix = "tournament_";
        if (maps.size() == 1) {
            Path mapPath = Paths.get(maps.get(0).m_a);
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
        final String traceOutputFolder = tournamentfolder + "/traces";
        final String folderForReadWriteFolders = tournamentfolder;

        final int playOnlyGamesInvolvingThisAI = 0;
        final int iterations = raiArgs.getOptionInteger('n', 1);
        final int iterationsBudget = -1;
        final int preAnalysisBudgetFirstTimeInAMap = 10000;
        final int preAnalysisBudgetRestOfTimes = 1000;
        final boolean fullObservability = true;
        final boolean selfMatches = false;
        final boolean runGC = true;
        final boolean preAnalysis = preAnalysisBudgetFirstTimeInAMap > 0;
        final Writer out = new FileWriter(fileToUse);
        final Writer progress = new PrintWriter(System.out);

        if (progress != null) {
            progress.write(tournament.getClass().getName() + ": Starting tournament\n");
            progress.write(
                    "overrideTorchThreads: " + overrideTorchThreads + "; pythonVerboseLevel: " + pythonVerboseLevel);
            if (useBestModels) {
                progress.write("; Use best models");
            }
            progress.write("\n");
        }
        out.write(tournament.getClass().getName() + "\n");
        out.write("AIs\n");
        for (AI ai : AIs) {
            out.write("\t" + ai.toString() + "\n");
        }
        out.write("maps\n");
        for (Pair<String, Integer> map : maps) {
            out.write("\t" + map.m_a + "\t" + map.m_b + "\n");
        }
        out.write("iterations\t" + iterations + "\n");
        out.write("timeBudget\t" + timeBudget + "\n");
        out.write("iterationsBudget\t" + iterationsBudget + "\n");
        out.write("pregameAnalysisBudget\t" + preAnalysisBudgetFirstTimeInAMap + "\t" + preAnalysisBudgetRestOfTimes
                + "\n");
        out.write("preAnalysis\t" + preAnalysis + "\n");
        out.write("fullObservability\t" + fullObservability + "\n");
        out.write("timeoutCheck\t" + timeoutCheck + "\n");
        out.write("runGC\t" + runGC + "\n");
        out.write("iteration\tmap\tai1\tai2\ttime\twinner\tcrashed\ttimedout\tai1time\tai2time\tai1over\tai2over\n");
        out.flush();

        // create all the read/write folders:
        String readWriteFolders[] = new String[AIs.length];
        for (int i = 0; i < AIs.length; i++) {
            readWriteFolders[i] = folderForReadWriteFolders + "/AI" + i + "readWriteFolder";
            File f = new File(readWriteFolders[i]);
            f.mkdir();
        }

        boolean firstPreAnalysis[][] = new boolean[AIs.length][maps.size()];
        for (int i = 0; i < AIs.length; i++) {
            for (int j = 0; j < maps.size(); j++) {
                firstPreAnalysis[i][j] = true;
            }
        }

        for (int iteration = 0; iteration < iterations; iteration++) {
            for (int map_idx = 0; map_idx < maps.size(); map_idx++) {
                var map = maps.get(map_idx);
                PhysicalGameState pgs = PhysicalGameState.load(map.m_a, utt);
                for (int ai1_idx = 0; ai1_idx < AIs.length; ai1_idx++) {
                    for (int ai2_idx = 0; ai2_idx < AIs.length; ai2_idx++) {
                        if (!selfMatches && ai1_idx == ai2_idx)
                            continue;
                        if (playOnlyGamesInvolvingThisAI != -1) {
                            if (ai1_idx != playOnlyGamesInvolvingThisAI &&
                                    ai2_idx != playOnlyGamesInvolvingThisAI)
                                continue;
                        }
                        progress.write("Starting iteration " + iteration +
                                " on " + map.m_a + "\n");
                        progress.flush();
                        tournament.playSingleGame(map.m_b, timeBudget, iterationsBudget,
                                preAnalysisBudgetFirstTimeInAMap, preAnalysisBudgetRestOfTimes, fullObservability,
                                timeoutCheck, runGC, preAnalysis, utt, traceOutputFolder, out, progress,
                                readWriteFolders, firstPreAnalysis, iteration, map_idx, pgs, ai1_idx, ai2_idx);
                    }
                }
            }
        }

        tournament.printEndSummary(maps.stream().map(p -> p.m_a).collect(Collectors.toList()), iterations, out,
                progress);
    }
}
