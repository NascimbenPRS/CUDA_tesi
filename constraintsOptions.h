#include "anyoption.h" // options parsing

// Read options from command line and update variables if needed
void readOptionsConstraints(int argc, char* argv[], int *usesCache, int *arraySize, int *a, int *b, int *c) {

	// parse options
	AnyOption *opt = new AnyOption();
	// set usage
	opt->addUsage("Options usage: ");
	opt->addUsage("");
	opt->addUsage(" --size <size>\tArray size (* 2^20) elements");
	opt->addUsage(" --a <a> \tNumber of 2-variables constraints ");
	opt->addUsage(" --b <b> \tNumber of 3-variables constraints ");
	opt->addUsage(" --c <c> \tNumber of [4..128]-variables constraints ");
	//opt->addUsage(" --no_cache \tDon't use cache ");
	opt->addUsage("");
	opt->printUsage();

	// set options
	opt->setOption("size");	
	opt->setOption("a");
	opt->setOption("b");
	opt->setOption("c");
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
			*arraySize = (1 << 20) * temp;
			printf("Array size set to: %dM integers\n", *arraySize);
		}	
	}
	if (opt->getValue("a") != NULL) {
		*a = atoi(opt->getValue("a"));
		printf("tNumber of 2-variables constraints set to: %d\n", *a);
	}
	if (opt->getValue("b") != NULL) {
		*b = atoi(opt->getValue("b"));
		printf("tNumber of 3-variables constraints set to: %d\n", *b);
	}
	if (opt->getValue("c") != NULL) {
		*a = atoi(opt->getValue("c"));
		printf("tNumber of [4..128]-variables constraints set to: %d\n", *c);
	}

	delete opt;
	// options parsed

}

