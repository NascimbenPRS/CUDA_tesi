#include "anyoption.h" // options parsing

// Read options from command line and update variables accordingly
void readOptions(int argc, char* argv[], int *usesCache, int *numCycles, int *numCyclesGPU, int *arraySize) {

	// parse options
	AnyOption *opt = new AnyOption();
	// set usage
	opt->addUsage("Options usage: ");
	opt->addUsage("");
	opt->addUsage(" --no_cache \tDon't use cache ");
	opt->addUsage(" --rep <rep>\tNumber of repetitions ");
	opt->addUsage(" --size <size>\tArray size (* 2^20) elements");
	opt->addUsage("");
	opt->printUsage();

	// set options
	opt->setFlag("no_cache");
	opt->setOption("rep");
	opt->setOption("size");

	// Process commandline and get the options
	opt->processCommandArgs(argc, argv);

	// Get option values

	if (opt->getFlag("no_cache")) {
		*usesCache = 0;
		printf("no_cache flag set\n");
	}
	if (opt->getValue("rep") != NULL) {
		*numCycles = atoi(opt->getValue("rep"));
		*numCyclesGPU = *numCycles;
		printf("Number of repetitions set to: %d\n", *numCycles);
	}
	if (opt->getValue("size") != NULL) {
		*arraySize = (1 << 20) * atoi(opt->getValue("size"));
		printf("Array size set to: %dM integers\n", *arraySize);
	}

	delete opt;
	// options parsed

}

