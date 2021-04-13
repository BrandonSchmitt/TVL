#include "tvl/AST.h"

#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace tvl::ast;

namespace {

	// RAII helper to manage increasing/decreasing the indentation as we traverse the AST
	struct Indent {
		Indent(int& level) : level(level) { ++level; }

		~Indent() { --level; }

		int& level;
	};

	/// Helper class that implement the AST tree traversal and print the nodes along the way. The only data member is
	/// the current indentation level.
	class ASTDumper {
	public:
		void dump(const Module* node);

	private:
		void dump(const Array* node);
		void dump(const ArrayIndexing* node);
		void dump(const Assignment* node);
		void dump(const BinaryOperator* node);
		void dump(const Declaration* node);
		void dump(const Expression* node);
		void dump(const ForLoop* node);
		void dump(const Function* node);
		void dump(const FunctionCall* node);
		void dump(const Identifier* node);
		void dump(const Number* node);
		void dump(const Range* node);
		void dump(const Statement* node);
		void dump(const StatementList* node);
		//void dump(VariableExprAST* node);
		//void dump(PrintExprAST* node);
		//void dump(PrototypeAST* node);

		// Actually print spaces matching the current indentation level
		void indent() const {
			for (int i = 0; i < curIndent; i++) {
				llvm::errs() << "|   ";
			}
		}

		int curIndent = 0;
	};

} // namespace

/// Return a formatted string for the location of any node
template<typename T>
static std::string loc(const T* node) {
	const auto& loc = node->getLocation();
	return (llvm::Twine("@") + loc.begin.filename + ":" + llvm::Twine(loc.begin.line) + ":" +
			llvm::Twine(loc.begin.column)).str();
}

// Helper Macro to bump the indentation level and print the leading spaces for
// the current indentations
#define INDENT()                                                               \
  Indent level_(curIndent);                                                    \
  indent();


void ASTDumper::dump(const Array* node) {
	INDENT();
	llvm::errs() << "Array (" << node->getEmittingLangType() << "): [ " << loc(node) << "\n";
	for (auto& expr : node->getElements()) {
		dump(expr.get());
	}
	indent();
	llvm::errs() << "]\n";
}

void ASTDumper::dump(const ArrayIndexing* node) {
	INDENT();
	llvm::errs() << "Array Indexing (" << node->getEmittingLangType() << "): (array, index) = ( " << loc(node) << "\n";
	dump(node->getArray().get());
	dump(node->getIndex().get());
	indent();
	llvm::errs() << ")\n";
}

void ASTDumper::dump(const Assignment* node) {
	INDENT();
	llvm::errs() << "Assignment (" << node->getEmittingLangType() << "): (place, value) = ( '" << loc(node) << "\n";
	dump(node->getPlace().get());
	dump(node->getValue().get());
	indent();
	llvm::errs() << ") // Assignment\n";
}

void ASTDumper::dump(const BinaryOperator* node) {
	INDENT();
	llvm::errs() << "Binary Operator (" << node->getEmittingLangType() << ") ";
	switch (node->getOperatorType()) {
		case BinaryOperator::Addition:
			llvm::errs() << "+";
			break;
		case BinaryOperator::Subtraction:
			llvm::errs() << "-";
			break;
		case BinaryOperator::Multiplication:
			llvm::errs() << "*";
			break;
		case BinaryOperator::Division:
			llvm::errs() << "/";
			break;
		case BinaryOperator::Remainder:
			llvm::errs() << "%";
			break;
	}
	llvm::errs() << " { " << loc(node) << "\n";
	dump(node->getLhs().get());
	dump(node->getRhs().get());
	indent();
	llvm::errs() << "} // Binary Operator\n";
}

void ASTDumper::dump(const Declaration* node) {
	INDENT();
	llvm::errs() << "Variable Declaration; Type '" << node->getTypeIdentifier() << "', Name '" << node->getName()
			<< "', Init: { " << loc(node) << "\n";
	dump(node->getExpression().get());
	indent();
	llvm::errs() << "} // Variable Declaration '" << node->getName() << "'\n";
}

/// Dispatch to a generic expressions to the appropriate subclass using RTTI
void ASTDumper::dump(const Expression* expr) {
	llvm::TypeSwitch<const Expression*>(expr)
			.Case<Array, ArrayIndexing, Assignment, BinaryOperator, FunctionCall, Identifier, Number, Range>(
					[&](auto* node) { this->dump(node); })
			.Default([&](const Expression*) {
				// No match, fallback to a generic message
				INDENT();
				llvm::errs() << "<unknown Expression, kind " << expr->getType() << ">\n";
			});
	static_assert(EXPRESSIONS_END - EXPRESSIONS_BEGIN == 8, "Not all expressions covered in ASTDumper.");
}

void ASTDumper::dump(const ForLoop* node) {
	INDENT();
	llvm::errs() << "For Loop, loop variable '" << node->getLoopVariable() << "', (iterable, body) = ( " << loc(node)
			<< "\n";
	dump(node->getIterable().get());
	dump(&node->getBody());
	indent();
	llvm::errs() << ") // For Loop\n";
}

/// Print a function, first the prototype and then the body.
void ASTDumper::dump(const Function* node) {
	INDENT();
	llvm::errs() << "Function '" << node->getIdentifier() << "' " << loc(node) << "\n";
	/*indent();
	llvm::errs() << "Params: [";
	llvm::errs() << "]\n";*/
	dump(&node->getBody());
}

/// Print a call expression, first the callee name and the list of args by recursing into each individual argument.
void ASTDumper::dump(const FunctionCall* node) {
	INDENT();
	llvm::errs() << "Call '" << node->getCallee() << "' [ " << loc(node) << "\n";
	for (auto& arg : node->getArguments()) {
		dump(arg.get());
	}
	indent();
	llvm::errs() << "]\n";
}

/// Print a variable reference (just a name).
void ASTDumper::dump(const Identifier* node) {
	INDENT();
	llvm::errs() << "var (" << node->getEmittingLangType() << "): " << node->getName() << " " << loc(node) << "\n";
}

/// A literal number, just print the value.
void ASTDumper::dump(const Number* node) {
	INDENT();
	llvm::errs() << node->getValue() << " (" << node->getEmittingLangType() << ") " << loc(node) << "\n";
}

void ASTDumper::dump(const Range* node) {
	INDENT();
	llvm::errs() << "Range (start end) = ( " << loc(node) << "\n";
	dump(node->getBegin().get());
	dump(node->getEnd().get());
	indent();
	llvm::errs() << ") // Range\n";
}

void ASTDumper::dump(const Statement* statement) {
	llvm::TypeSwitch<const Statement*>(statement)
			.Case<Declaration, Expression, ForLoop>([&](auto* node) { this->dump(node); })
			.Default([&](const Statement*) {
				INDENT();
				llvm::errs() << "<unknown Statement, kind " << statement->getType() << ">\n";
			});
	static_assert(STATEMENTS_END - STATEMENTS_BEGIN - (EXPRESSIONS_END - EXPRESSIONS_BEGIN) -
			(DECLARATIONS_END - DECLARATIONS_BEGIN) == 1, "Not all statements covered in ASTDumper.");
}

/// A "block", or a list of expression
void ASTDumper::dump(const StatementList* statementList) {
	INDENT();
	llvm::errs() << "Block {\n";
	for (auto& stmt : *statementList) {
		dump(stmt.get());
	}
	indent();
	llvm::errs() << "} // Block\n";
}

/*
/// Print a builtin print call, first the builtin name and then the argument.
void ASTDumper::dump(PrintExprAST* node) {
	INDENT();
	llvm::errs() << "Print [ " << loc(node) << "\n";
	dump(node->getArg());
	indent();
	llvm::errs() << "]\n";
}
 */
/*
/// Print a function prototype, first the function name, and then the list of
/// parameters names.
void ASTDumper::dump(PrototypeAST* node) {
	INDENT();
	llvm::errs() << "Proto '" << node->getName() << "' " << loc(node) << "\n";
	indent();
	llvm::errs() << "Params: [";
	llvm::errs() << "]\n";
}*/

/// Print a module, actually loop over the functions and print them in sequence.
void ASTDumper::dump(const Module* node) {
	llvm::errs() << "Module:\n";
	for (auto& f : *node) {
		dump(f.get());
	}
}

namespace tvl {
	namespace ast {
		void dump(const Module& module) { ASTDumper().dump(&module); }
	}    // namespace ast
}    // namespace tvl
