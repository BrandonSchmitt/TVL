#ifndef TVL_DIALECT_PARSINGDRIVER_H
#define TVL_DIALECT_PARSINGDRIVER_H

#include <memory>
#include "parser.h"
#include "tvl/AST.h"
#include "llvm/ADT/StringRef.h"

// Give Flex the prototype of yylex
#define YY_DECL \
tvl::Parser::symbol_type yylex(tvl::ParsingDriver& driver, tvl::Parser* parser)
YY_DECL;    // Define the function for the parser

namespace tvl {
	namespace ast {
		class Function;
	}
	class ParsingDriver {
	public:
		ParsingDriver() : traceScanning{false}, traceParsing{false} {};

		int parse(llvm::StringRef filename, llvm::StringRef source);
		void addFunction(std::unique_ptr<ast::Function> function) { module->addFunction(std::move(function)); }
		std::unique_ptr<ast::Module> getModule() { auto m = std::move(module); module = nullptr; return m; }

		bool traceScanning;
		bool traceParsing;
		tvl::ast::Location location;

	private:
		void beginScan(llvm::StringRef source);
		void endScan();

		std::unique_ptr<ast::Module> module;
	};
}

#endif //TVL_DIALECT_PARSINGDRIVER_H
