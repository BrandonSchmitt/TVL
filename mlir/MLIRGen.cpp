#include "tvl/MLIRGen.h"
#include "tvl/AST.h"
#include "tvl/TvlDialect.h"
#include "tvl/TvlOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>


// Todo: Maybe replace it
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir::tvl;
using namespace tvl;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

	/// Implementation of a simple MLIR emission from the TVL AST.
	///
	/// This will emit operations that are specific to the TV, preserving the semantics of the language and (hopefully)
	/// allow to perform accurate analysis and transformation based on these high level semantics.
	class MLIRGenImpl {
	public:
		MLIRGenImpl(mlir::MLIRContext& context) : builder(&context) {}

		/// Public API: convert the AST for a TVL module (source file) to an MLIR Module operation.
		mlir::ModuleOp mlirGen(ast::Module& moduleAST) {
			// We create an empty MLIR module and codegen functions one at a time and add them to the module.
			theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

			for (auto& functionAST : moduleAST) {
				auto func = mlirGen(*functionAST);
				if (!func) {
					return nullptr;
				}
				theModule.push_back(func);
			}

			// Verify the module after we have finished constructing it, this will check the structural properties of
			// the IR and invoke any specific verifiers we have on the TVL operations.
			if (failed(mlir::verify(theModule))) {
				theModule.emitError("module verification error");
				return nullptr;
			}

			return theModule;
		}

	private:
		/// A "module" matches a TVL source file: containing a list of functions.
		mlir::ModuleOp theModule;

		/// The builder is a helper class to create IR inside a function. The builder is stateful, in particular it
		/// keeps an "insertion point": this is where the next operations will be introduced.
		mlir::OpBuilder builder;

		/// The symbol table maps a variable name to a value in the current scope. Entering a function creates a new
		/// scope, and the function arguments are added to the mapping. When the processing of a function is terminated,
		/// the scope is destroyed and the mappings created in this scope are dropped.
		llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;

		/// Helper conversion for a TVL AST location to an MLIR location.
		mlir::Location loc(const ast::Location& loc) {
			return mlir::FileLineColLoc::get(builder.getIdentifier(loc.begin.filename), loc.begin.line,
					loc.begin.column);
		}

		/// Declare a variable in the current scope, return success if the variable wasn't declared yet.
		mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
			if (symbolTable.count(var)) {
				return mlir::failure();
			}
			symbolTable.insert(var, value);
			return mlir::success();
		}

		mlir::LogicalResult assign(llvm::StringRef var, mlir::Value value) {
			if (symbolTable.count(var)) {
				symbolTable.insert(var, value);
				return mlir::success();
			}
			return mlir::failure();
		}

		mlir::Value mlirGen(const ast::Array& node) {
			auto location = loc(node.getLocation());

			auto integerType = builder.getI64Type();
			auto arrayLength = node.getElements().size();
			std::vector<int64_t> shape{static_cast<int64_t>(arrayLength)};
			auto memRefType = mlir::MemRefType::get(shape, integerType);
			auto memRef = builder.create<mlir::memref::AllocaOp>(location, memRefType);

			for (size_t i = 0; i < arrayLength; ++i) {
				mlir::Value index = builder.create<ConstantOp>(location, builder.getIndexAttr(i));
				builder.create<mlir::memref::StoreOp>(location, mlirGen(*node.getElements().at(i)), memRef,
						mlir::ValueRange(index));
			}
			return memRef;
		}

		mlir::Value mlirGen(const ast::ArrayIndexing& node) {
			return builder.create<LoadOp>(loc(node.getLocation()), mlirGen(*node.getArray()),
					mlirGen(*node.getIndex()));
		}

		mlir::Value mlirGen(const ast::Assignment& node) {
			auto& valueExpression = node.getValue();
			if (!valueExpression) {
				emitError(loc(node.getLocation()), "missing value in variable assignment");
				return nullptr;
			}

			mlir::Value value = mlirGen(*valueExpression);
			if (!value) {
				return nullptr;
			}

			// Register the value in the symbol table.
			switch (node.getPlace()->getType()) {
				case ast::ArrayIndexingNode: {
					auto& place = cast<ast::ArrayIndexing>(*node.getPlace());
					auto memRef = mlirGen(*place.getArray());
					auto index = mlirGen(*place.getIndex());
					builder.create<StoreOp>(loc(node.getLocation()), value, memRef, index);
					break;
				}
				case ast::IdentifierNode:
					if (failed(assign(cast<ast::Identifier>(*node.getPlace()).getName(), value))) {
						return nullptr;
					}
					break;
				default:
					return nullptr;
			}
			return value;
		}

		mlir::Value mlirGen(const ast::BinaryOperator& binaryOperator) {
			mlir::Value lhs = mlirGen(*binaryOperator.getLhs());
			if (!lhs) {
				return nullptr;
			}
			mlir::Value rhs = mlirGen(*binaryOperator.getRhs());
			if (!rhs) {
				return nullptr;
			}

			auto location = loc(binaryOperator.getLocation());

			switch (binaryOperator.getOperatorType()) {
				case ast::BinaryOperator::Addition:
					return builder.create<AddOp>(location, lhs, rhs);
				case ast::BinaryOperator::Subtraction:
					return builder.create<SubOp>(location, lhs, rhs);
				case ast::BinaryOperator::Multiplication:
					return builder.create<MulOp>(location, lhs, rhs);
				case ast::BinaryOperator::Division:
					return builder.create<DivOp>(location, lhs, rhs);
				case ast::BinaryOperator::Remainder:
					return builder.create<RemOp>(location, lhs, rhs);
			}
		}

		mlir::Value mlirGen(const ast::Declaration& node) {
			auto& init = node.getExpression();
			if (!init) {
				emitError(loc(node.getLocation()), "missing initializer in variable declaration");
				return nullptr;
			}

			mlir::Value value = mlirGen(*init);
			if (!value) {
				return nullptr;
			}

			// Register the value in the symbol table.
			if (failed(declare(node.getName(), value))) {
				return nullptr;
			}
			return value;
		}

		/// Dispatch codegen for the right expression subclass using RTTI.
		mlir::Value mlirGen(const ast::Expression& node) {
			switch (node.getType()) {
				case ast::ArrayNode:
					return mlirGen(cast<ast::Array>(node));
				case ast::ArrayIndexingNode:
					return mlirGen(cast<ast::ArrayIndexing>(node));
				case ast::AssignmentNode:
					return mlirGen(cast<ast::Assignment>(node));
				case ast::BinaryOperatorNode:
					return mlirGen(cast<ast::BinaryOperator>(node));
				case ast::FunctionCallNode:
					return mlirGen(cast<ast::FunctionCall>(node));
				case ast::IdentifierNode:
					return mlirGen(cast<ast::Identifier>(node));
				case ast::IntegerNode:
					return mlirGen(cast<ast::Integer>(node));
				case ast::RangeNode:
					emitError(loc(node.getLocation())) << "MLIR codegen does not yet support range expressions.";
					return nullptr;
					//return mlirGen(cast<ast::Range(node));
				default:
					emitError(loc(node.getLocation())) << "MLIR codegen encountered an unhandled expression kind '"
							<< Twine(node.getType()) << "'";
					return nullptr;
			}
			static_assert(ast::EXPRESSIONS_END - ast::EXPRESSIONS_BEGIN == 8,
					"Not all expressions covered in MLIRGen.");
		}

		mlir::LogicalResult mlirGen(const ast::ForLoop& node) {
			auto& iterable = node.getIterable();
			if (!isa<ast::Range>(*iterable)) {
				emitError(loc(node.getLocation()), "error: only ranges are supported by for loops at the moment");
				return mlir::failure();
			}
			auto& range = cast<ast::Range>(*iterable);

			auto location = loc(node.getLocation());

			ForOp forOp = builder.create<ForOp>(location, llvm::None, mlirGen(*range.getBegin()),
					mlirGen(*range.getEnd()));
			auto& region = forOp.region();
			region.push_back(new mlir::Block());
			auto& body = region.front();
			body.addArgument(builder.getIndexType());

			mlir::OpBuilder::InsertionGuard guard(builder);
			builder.setInsertionPointToStart(&body);

			ScopedHashTableScope<StringRef, mlir::Value> var_scope(symbolTable);
			declare(node.getLoopVariable(), body.getArgument(0));

			if (failed(mlirGen(node.getBody()))) {
				return mlir::failure();
			}

			builder.create<YieldOp>(location);
			return mlir::success();
		}

		/// Emit a new function and add it to the MLIR module.
		mlir::FuncOp mlirGen(const ast::Function& funcAST) {
			// Create a scope in the symbol table to hold variable declarations.
			ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(symbolTable);

			// Create an MLIR function for the given prototype.
			// mlir::FuncOp function(mlirGen(*funcAST.getProto()));
			auto location = loc(funcAST.getLocation());

			// This is a generic function, the return type will be inferred later. Arguments type are uniformly
			// uint64_t.
			auto func_type = builder.getFunctionType(llvm::None, llvm::None);
			auto function = mlir::FuncOp::create(location, funcAST.getIdentifier(), func_type);

			if (!function) {
				return nullptr;
			}

			// Let's start the body of the function now! In MLIR the entry block of the function is special: it must
			// have the same argument list as the function itself.
			auto& entryBlock = *function.addEntryBlock();

			// Set the insertion point in the builder to the beginning of the function body, it will be used throughout
			// the codegen to create operations in this function.
			builder.setInsertionPointToStart(&entryBlock);

			// Emit the body of the function.
			if (mlir::failed(mlirGen(funcAST.getBody()))) {
				function.erase();
				return nullptr;
			}

			ReturnOp returnOp;
			if (!entryBlock.empty()) {
				returnOp = dyn_cast<ReturnOp>(entryBlock.back());
			}
			if (!returnOp) {
				builder.create<ReturnOp>(loc(funcAST.getLocation()));
			} else if (returnOp.hasOperand()) {
				//function.setType(builder.getFunctionType(function.getType().getInputs(), builder.getI64Type()));
			}

			return function;
		}

		/// Emit a call expression. It emits specific operations for the builtins. Other identifiers are
		/// assumed to be user-defined functions.
		mlir::Value mlirGen(const ast::FunctionCall& call) {
			llvm::StringRef callee = call.getCallee();
			auto location = loc(call.getLocation());

			// Codegen the operands first.
			SmallVector<mlir::Value, 4> operands;
			for (auto& expr : call.getArguments()) {
				auto arg = mlirGen(*expr);
				if (!arg) {
					return nullptr;
				}
				operands.push_back(arg);
			}

			// Otherwise this is a call to a user-defined function. Calls to user-defined functions are mapped to a
			// custom call that takes the callee name as an attribute.
			return builder.create<GenericCallOp>(location, callee, operands);
		}

		mlir::LogicalResult mlirGenPrint(const ast::FunctionCall& call) {
			if (call.getArguments().size() != 1) {
				return mlir::failure();
			}
			auto arg = mlirGen(*call.getArguments().front());
			if (!arg) {
				return mlir::failure();
			}
			builder.create<PrintOp>(loc(call.getLocation()), arg);
			return mlir::success();
		}

		/// This is a reference to a variable in an expression. The variable is expected to have been declared and so
		/// should have a value in the symbol table, otherwise emit an error and return nullptr.
		mlir::Value mlirGen(const ast::Identifier& node) {
			if (auto variable = symbolTable.lookup(node.getName())) {
				return variable;
			}

			emitError(loc(node.getLocation()), "unknown identifier '") << node.getName() << "'";
			return nullptr;
		}

		/// Emit a constant for a single number (FIXME: semantic? broadcast?)
		mlir::Value mlirGen(const ast::Integer& num) {
			mlir::Attribute attribute;
			switch (num.getEmittingLangType().baseType) {
				case u64:
					attribute = builder.getI64IntegerAttr(num.getValue());
					break;
				case u32:
					attribute = builder.getI32IntegerAttr(num.getValue());
					break;
				case u16:
					attribute = builder.getI16IntegerAttr(num.getValue());
					break;
				case u8:
					attribute = builder.getI8IntegerAttr(num.getValue());
					break;
				case usize:
					attribute = builder.getIndexAttr(num.getValue());
					break;
				case f64:
					attribute = builder.getF64FloatAttr(num.getValue());
					break;
				case f32:
					attribute = builder.getF32FloatAttr(num.getValue());
					break;
				default:
					emitError(loc(num.getLocation())) << "Unknown integer type: "
							<< static_cast<int>(num.getEmittingLangType().baseType);
					return nullptr;
			}
			return builder.create<ConstantOp>(loc(num.getLocation()), attribute);
		}

		mlir::LogicalResult mlirGen(const ast::Statement& statement) {
			if (isa<ast::Expression>(statement)) {
				return mlir::success(mlirGen(cast<ast::Expression>(statement)) != nullptr);
			}
			if (isa<ast::Declaration>(statement)) {
				return mlir::success(mlirGen(cast<ast::Declaration>(statement)) != nullptr);
			}
			switch (statement.getType()) {
				case ast::ForLoopNode:
					return mlirGen(cast<ast::ForLoop>(statement));
				default:
					emitError(loc(statement.getLocation())) << "MLIR codegen encountered an unhandled statement kind '"
							<< Twine(statement.getType()) << "'";
					return mlir::failure();
			}
			static_assert(
					ast::STATEMENTS_END - ast::STATEMENTS_BEGIN - (ast::EXPRESSIONS_END - ast::EXPRESSIONS_BEGIN) -
							(ast::DECLARATIONS_END - ast::DECLARATIONS_BEGIN) == 1,
					"Not all statements covered in MLIRGen.");
		}

		/// Codegen a list of expression, return failure if one of them hit an error.
		mlir::LogicalResult mlirGen(const ast::StatementList& blockAST) {
			ScopedHashTableScope<StringRef, mlir::Value> var_scope(symbolTable);
			for (auto& stmt : blockAST) {
				// Specific handling for variable declarations, return statement, and print. These can only appear in
				// block list and not in nested expressions.
				if (auto* func = dyn_cast<ast::FunctionCall>(stmt.get())) {
					if (func->getCallee() == "print") {
						if (mlir::failed(mlirGenPrint(*func))) {
							return mlir::failure();
						}
						continue;
					}
				}

				// Generic expression dispatch codegen.
				if (mlir::failed(mlirGen(*stmt))) {
					return mlir::failure();
				}
			}
			return mlir::success();
		}
	};

} // namespace

namespace tvl {

	// The public API for codegen.
	mlir::OwningModuleRef mlirGen(mlir::MLIRContext& context, ast::Module& moduleAST) {
		return MLIRGenImpl(context).mlirGen(moduleAST);
	}

} // namespace tvl
