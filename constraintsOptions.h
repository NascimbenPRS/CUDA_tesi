#include "anyoption.h" // options parsing

// Read options from command line and update variables if needed
void readOptionsConstraints(int argc, char* argv[], int *usesCache, int *arraySize, int *a, int *b, int *c, int *seed, int *numRep) {

	// parse options
	AnyOption *opt = new AnyOption();
	// set usage
	opt->addUsage("Options usage: ");
	opt->addUsage("");
	opt->addUsage(" --size <size>\tArray size ");
	opt->addUsage(" --a <a> \tNumber of 2-variables constraints ");
	opt->addUsage(" --b <b> \tNumber of 3-variables constraints ");
	opt->addUsage(" --c <c> \tNumber of [4..128]-variables constraints ");
	opt->addUsage(" --seed <seed> \tSeed for the random functions ");
	opt->addUsage(" --numRep <numRep> \tNumber of repetitions ");
	//opt->addUsage(" --no_cache \tDon't use cache ");
	opt->addUsage("");
	opt->printUsage();

	// set options
	opt->setOption("size");	
	opt->setOption("a");
	opt->setOption("b");
	opt->setOption("c");
	opt->setOption("seed");
	opt->setOption("numRep");
	//opt->setFlag("no_cache");


	// Process commandline and get the options
	opt->processCommandArgs(argc, argv);

	// Get option values
	/*
	if (opt->getFlag("no_cache")) {
		*usesCache = 0;
		printf("no_cache flag set\n");
	}
	*/

	if (opt->getValue("size") != NULL) {
		int temp = atoi(opt->getValue("size"));
		if (temp > 0) {
			*arraySize = temp;
			printf("Array size set to: %d integers\n", *arraySize);
		}	
	}
	if (opt->getValue("a") != NULL) {
		*a = atoi(opt->getValue("a"));
		printf("Number of 2-variables constraints set to: %d\n", *a);
	}
	if (opt->getValue("b") != NULL) {
		*b = atoi(opt->getValue("b"));
		printf("Number of 3-variables constraints set to: %d\n", *b);
	}
	if (opt->getValue("c") != NULL) {
		*c = atoi(opt->getValue("c"));
		printf("Number of [4..128]-variables constraints set to: %d\n", *c);
	}

	if (opt->getValue("seed") != NULL) {
		*seed = atoi(opt->getValue("seed"));
		printf("Seed set to: %d\n", *seed);
	}
	if (opt->getValue("numRep") != NULL) {
		*numRep = atoi(opt->getValue("numRep"));
		printf("Number of repetitions set to: %d\n", *numRep);
	}


	delete opt;
	// options parsed

}

