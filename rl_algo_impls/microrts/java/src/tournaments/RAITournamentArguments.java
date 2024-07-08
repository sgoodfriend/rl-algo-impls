package tournaments;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public class RAITournamentArguments {
    private CommandLine cmd;
    public int pythonVerbosity = 1;

    public RAITournamentArguments(String args[]) throws ParseException {
        Options options = new Options();
        options.addOption("n", "num-iterations", true, "Number of iterations to play round-robin tournament");
        options.addOption("t", "time-budget", true, "Milliseconds each turn is allowed to have");
        options.addOption("T", "no-timeout", false,
                "Losses aren't declared for timeouts, but AIs still passed time-budget");
        options.addOption("b", "use-best-models", false,
                "Disable performance-based model selection in RAISocketAI. Always pick highest precedence model");
        options.addOption("p", "override-torch-threads", true,
                "Override torch threads to this value. Ignoring other logic");
        options.addOption("v", "python-verbose", false, "Make Python process logging extra verbose");
        options.addOption("q", "quiet", false, "Make Python process not log to file");
        options.addOption("m", "model-set", true, "Use the specified model set");
        CommandLineParser parser = new DefaultParser();
        try {
            cmd = parser.parse(options, args);
        } catch (ParseException exp) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp(
                    StackWalker.getInstance(StackWalker.Option.RETAIN_CLASS_REFERENCE).getCallerClass().getSimpleName(),
                    options);
            throw exp;
        }
        if (cmd.hasOption('q')) {
            pythonVerbosity = 0;
        } else if (cmd.hasOption('v')) {
            pythonVerbosity = 2;
        }
    }

    public boolean hasOption(char opt) {
        return cmd.hasOption(opt);
    }

    public int getOptionInteger(char opt, int defaultValue) {
        if (cmd.hasOption(opt)) {
            return Integer.valueOf(cmd.getOptionValue(opt));
        } else {
            return defaultValue;
        }
    }

    public String getOptionValue(char opt, String defaultValue) {
        if (cmd.hasOption(opt)) {
            return cmd.getOptionValue(opt);
        } else {
            return defaultValue;
        }
    }
}
