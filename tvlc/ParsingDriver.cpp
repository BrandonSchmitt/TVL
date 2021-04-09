#include "tvl/ParsingDriver.h"

int tvl::ParsingDriver::parse(llvm::StringRef filename, llvm::StringRef source) {
	location.initialize(filename);
	module = std::make_unique<tvl::ast::Module>(location);

	beginScan(source);

	tvl::Parser parser(*this);
	parser.set_debug_level(traceParsing);
	int result = parser();

	endScan();

	return result;
}
