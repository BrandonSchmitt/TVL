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
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"


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

			auto integerType = getMlirType(node.getEmittingLangType().elementType->baseType);
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
				emitError(loc(node.getLocation()), "There is already a variable with the name " + node.getName());
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
				case ast::StringNode:
					return mlirGen(cast<ast::String>(node));
				default:
					emitError(loc(node.getLocation())) << "MLIR codegen encountered an unhandled expression kind '"
							<< Twine(node.getType()) << "'";
					return nullptr;
			}
			static_assert(ast::EXPRESSIONS_END - ast::EXPRESSIONS_BEGIN == 9,
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

			if (callee == "vecAdd") {
				return mlirGenBinaryOperator<AddOp>(call);
			}
			if (callee == "vecBroadcast") {
				return mlirGenVectorBroadcast(call);
			}
			if (callee == "vecDiv") {
				return mlirGenBinaryOperator<DivOp>(call);
			}
			if (callee == "vecExtractElement") {
				return mlirGenVectorExtractElement(call);
			}
			if (callee == "vecGather") {
				return mlirGenVectorGather(call);
			}
			if (callee == "vecHAdd") {
				return mlirGenVectorHAdd(call);
			}
			if (callee == "vecLoad") {
				return mlirGenVectorLoad(call);
			}
			if (callee == "vecMul") {
				return mlirGenBinaryOperator<MulOp>(call);
			}
			if (callee == "vecRem") {
				return mlirGenBinaryOperator<RemOp>(call);
			}
			if (callee == "vecSeq") {
				return mlirGenVectorSeq(call);
			}
			if (callee == "vecShiftLeft") {
				return mlirGenShiftOperator<ShiftLeftOp>(call);
			}
			if (callee == "vecShiftLeftIndividual") {
				return mlirGenBinaryOperator<ShiftLeftOp>(call);
			}
			if (callee == "vecShiftRightUnsigned") {
				return mlirGenShiftOperator<ShiftRightUnsignedOp>(call);
			}
			if (callee == "vecShiftRightUnsignedIndividual") {
				return mlirGenBinaryOperator<ShiftRightUnsignedOp>(call);
			}
			if (callee == "vecShiftRightSigned") {
				return mlirGenShiftOperator<ShiftRightSignedOp>(call);
			}
			if (callee == "vecShiftRightSignedIndividual") {
				return mlirGenBinaryOperator<ShiftRightSignedOp>(call);
			}
			if (callee == "vecSub") {
				return mlirGenBinaryOperator<SubOp>(call);
			}
			if (callee == "rand_u64") {
				return mlirGenRand_u64(call);
			}

			if (callee == "maskInit") {
				return mlirGenMaskInit(call);
			}
			if (callee == "maskAnd" || callee == "vecAnd") {
				return mlirGenBinaryOperator<AndOp>(call);
			}
			if (callee == "maskOr" || callee == "vecOr") {
				return mlirGenBinaryOperator<OrOp>(call);
			}
			if (callee == "maskXOr" || callee == "vecXOr") {
				return mlirGenBinaryOperator<XOrOp>(call);
			}
			if (callee == "maskCountTrue") {
				return mlirGenUnaryOperator<MaskCountTrueOp>(call);
			}
			if (callee == "maskCountFalse") {
				return mlirGenUnaryOperator<MaskCountFalseOp>(call);
			}
			if (callee == "vecMin") {
				return mlirGenBinaryOperator<MinOp>(call);
			}
			if (callee == "vecMax") {
				return mlirGenBinaryOperator<MaxOp>(call);
			}
			if (callee == "vecEq") {
				return mlirGenBinaryCmpOperator<EqOp>(call);
			}
			if (callee == "vecNe") {
				return mlirGenBinaryCmpOperator<NeOp>(call);
			}
			if (callee == "vecGe") {
				return mlirGenBinaryCmpOperator<UgeOp>(call);
			}
			if (callee == "vecGt") {
				return mlirGenBinaryCmpOperator<UgtOp>(call);
			}
			if (callee == "vecLe") {
				return mlirGenBinaryCmpOperator<UleOp>(call);
			}
			if (callee == "vecLt") {
				return mlirGenBinaryCmpOperator<UltOp>(call);
			}

			if (callee == "instantNow") {
				return mlirGenInstantNow(call);
			}
			if (callee == "instantElapsed") {
				return mlirGenInstantElapsed(call);
			}

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

		mlir::Value mlirGen(const ast::String& string) {
			return getOrCreateGlobalString(loc(string.getLocation()), string.getString());
		}

		template<typename Op>
		mlir::Value mlirGenUnaryOperator(const ast::FunctionCall& call) {
			if (call.getArguments().size() != 1) {
				return nullptr;
			}
			auto operand = mlirGen(*call.getArguments().front());
			if (!operand) {
				return nullptr;
			}
			return builder.create<Op>(loc(call.getLocation()), operand);
		}

		template<typename Op>
		mlir::Value mlirGenBinaryOperator(const ast::FunctionCall& call) {
			if (call.getArguments().size() != 2) {
				return nullptr;
			}
			auto lhs = mlirGen(*call.getArguments().front());
			if (!lhs) {
				return nullptr;
			}
			auto rhs = mlirGen(*call.getArguments().back());
			if (!rhs) {
				return nullptr;
			}
			return builder.create<Op>(loc(call.getLocation()), lhs, rhs);
		}

		template<typename Op>
		mlir::Value mlirGenBinaryCmpOperator(const ast::FunctionCall& call) {
			if (call.getArguments().size() != 2) {
				return nullptr;
			}
			auto lhs = mlirGen(*call.getArguments().front());
			if (!lhs) {
				return nullptr;
			}
			auto rhs = mlirGen(*call.getArguments().back());
			if (!rhs) {
				return nullptr;
			}

			return builder.create<Op>(loc(call.getLocation()), lhs, rhs);
		}

		template<typename ShiftOp>
		mlir::Value mlirGenShiftOperator(const ast::FunctionCall& call) {
			if (call.getArguments().size() != 2) {
				return nullptr;
			}

			auto vector = mlirGen(*call.getArguments().front());
			if (!vector) {
				return nullptr;
			}
			auto vectorType = vector.getType().dyn_cast<mlir::VectorType>();
			if (!vectorType) {
				return nullptr;
			}

			auto bits = mlirGen(*call.getArguments().back());
			if (!bits) {
				return nullptr;
			}

			mlir::Value broadcast = builder.create<VectorBroadcastOp>(loc(call.getLocation()), vectorType, bits);
			return builder.create<ShiftOp>(loc(call.getLocation()), vector, broadcast);
		}

		mlir::Value mlirGenInstantNow(const ast::FunctionCall& call) {
			ArrayRef<mlir::Type> elementTypes{builder.getI64Type(), builder.getI64Type()};
			return builder.create<InstantNowOp>(loc(call.getLocation()), builder.getType<StructType>(elementTypes));
		}

		mlir::Value mlirGenInstantElapsed(const ast::FunctionCall& call) {
			if (call.getArguments().size() != 1) {
				return nullptr;
			}

			auto instant = mlirGen(*call.getArguments().front());
			if (!instant) {
				return nullptr;
			}

			return builder.create<InstantElapsedOp>(loc(call.getLocation()), builder.getF64Type(), instant);
		}

		mlir::Value mlirGenMaskInit(const ast::FunctionCall& call) {
			if (call.getTemplateArguments().size() != 1) {
				return nullptr;
			}
			auto length = std::get<ast::IntegerPtr>(call.getTemplateArgument(0))->getValue();

			auto location = loc(call.getLocation());
			auto value = builder.create<ConstantOp>(location, builder.getIntegerAttr(builder.getI1Type(), 0));
			return builder.create<VectorBroadcastOp>(location, mlir::VectorType::get(length, builder.getI1Type()), value);
		}

		mlir::LogicalResult mlirGenPrint(const ast::FunctionCall& call) {
			if (call.getArguments().empty()) {
				return mlir::failure();
			}
			if (call.getArguments().front()->getType() != tvl::ast::StringNode) {
				return mlir::failure();
			}

			StringRef formatStr = dyn_cast<ast::String>(call.getArguments().front().get())->getString();
			SmallVector<StringRef, 4> parts;
			formatStr.split(parts, "{}");

			if (parts.size() != call.getArguments().size()) {
				llvm::errs() << loc(call.getLocation()) << ": print expects " << parts.size() - 1 << " arguments, " << call.getArguments().size() - 1 << " given.";
				return mlir::failure();
			}

			llvm::SmallString<64> printfStr = parts[0];
			for (size_t i = 1, len = parts.size(); i < len; ++i) {
				switch (call.getArguments().at(i)->getEmittingLangType().baseType) {
					case u8:
						printfStr += "%hhu";
						static_assert(sizeof(unsigned char) == 1, "%hhu is the wrong identifier for printf");
						break;
					case u16:
						printfStr += "%hu";
						static_assert(sizeof(unsigned short int) == 2, "%hu is the wrong identifier for printf");
						break;
					case u32:
						printfStr += "%u";
						static_assert(sizeof(unsigned int) == 4, "%u is the wrong identifier for printf");
						break;
					case u64:
						printfStr += "%lu";
						static_assert(sizeof(unsigned long int) == 8, "%lu is the wrong identifier for printf");
						break;
					case usize:
						printfStr += "%zu";
						static_assert(sizeof(size_t) == 8, "%zu is the wrong identifier for printf");
						break;
					case i8:
						printfStr += "%hhi";
						static_assert(sizeof(char) == 1, "%hhi is the wrong identifier for printf");
						break;
					case i16:
						printfStr += "%hi";
						static_assert(sizeof(short int) == 2, "%hi is the wrong identifier for printf");
						break;
					case i32:
						printfStr += "%i";
						static_assert(sizeof(int) == 4, "%i is the wrong identifier for printf");
						break;
					case i64:
						printfStr += "%li";
						static_assert(sizeof(long int) == 8, "%li is the wrong identifier for printf");
						break;
					/*case f32: Need to promote f32 to f64 first. printf does not cover f32.
						printfStr += "%f";
						static_assert(sizeof(double) == 4, "%f is the wrong identifier for printf");
						break;*/
					case f64:
						printfStr += "%f";
						static_assert(sizeof(double) == 8, "%f is the wrong identifier for printf");
						break;
					case string:
						printfStr += "%s";
						break;
					default:
						return mlir::failure();
				}
				printfStr += parts[i];
			}
			printfStr += StringRef("\n\0", 2);

			SmallVector<mlir::Value, 4> operands;
			for (size_t i = 1, len = parts.size(); i < len; ++i) {
				auto arg = mlirGen(*call.getArguments().at(i));
				if (!arg) {
					return mlir::failure();
				}
				operands.push_back(arg);
			}

			auto location = loc(call.getLocation());

			mlir::Value formatString = getOrCreateGlobalString(location, printfStr);
			builder.create<PrintOp>(location, formatString, operands);
			return mlir::success();
		}

		mlir::LogicalResult mlirGenSrand(const ast::FunctionCall& call) {
			if (call.getArguments().size() != 1) {
				return mlir::failure();
			}
			auto arg = mlirGen(*call.getArguments().front());
			if (!arg) {
				return mlir::failure();
			}
			builder.create<SRandOp>(loc(call.getLocation()), arg);
			return mlir::success();
		}

		mlir::Value mlirGenRand_u64(const ast::FunctionCall& call) {
			return builder.create<RandOp>(loc(call.getLocation()), builder.getI64Type());
		}

		mlir::Value mlirGenVectorBroadcast(const ast::FunctionCall& call) {
			if (call.getTemplateArguments().size() != 1 || call.getArguments().size() != 1) {
				return nullptr;
			}

			auto value = mlirGen(*call.getArguments().front());
			if (!value) {
				return nullptr;
			}

			auto length = std::get<ast::IntegerPtr>(call.getTemplateArgument(0))->getValue();

			return builder.create<VectorBroadcastOp>(loc(call.getLocation()),
					mlir::VectorType::get(length, value.getType()), value);
		}

		mlir::LogicalResult mlirGenVectorCompressStore(const ast::FunctionCall& call) {
			if (call.getArguments().size() != 3) {
				return mlir::failure();
			}

			auto vec = mlirGen(*call.getArguments().front());
			if (!vec) {
				return mlir::failure();
			}

			auto mask = mlirGen(*call.getArguments().at(1));
			if (!mask) {
				return mlir::failure();
			}

			auto mem = mlirGen(*call.getArguments().back());
			if (!mem) {
				return mlir::failure();
			}

			mlir::Value indexZero = builder.create<ConstantOp>(loc(call.getLocation()), builder.getIndexAttr(0));
			builder.create<VectorCompressStoreOp>(loc(call.getLocation()), vec, mask, mem, mlir::ValueRange({indexZero}));
			return mlir::success();
		}

		mlir::Value mlirGenVectorExtractElement(const ast::FunctionCall& call) {
			if (call.getArguments().size() != 2) {
				return nullptr;
			}

			auto vec = mlirGen(*call.getArguments().front());
			if (!vec) {
				return nullptr;
			}

			auto idx = mlirGen(*call.getArguments().back());
			if (!idx) {
				return nullptr;
			}

			return builder.create<VectorExtractElementOp>(loc(call.getLocation()), vec.getType().cast<mlir::VectorType>().getElementType(), vec, idx);
		}

		mlir::Value mlirGenVectorGather(const ast::FunctionCall& call) {
			if (call.getArguments().size() != 2) {
				return nullptr;
			}

			auto memRef = mlirGen(*call.getArguments().front());
			if (!memRef) {
				return nullptr;
			}
			auto memRefType = memRef.getType().dyn_cast<mlir::MemRefType>();
			if (!memRefType) {
				return nullptr;
			}

			auto indexVector = mlirGen(*call.getArguments().back());
			if (!indexVector) {
				return nullptr;
			}
			auto indexVectorType = indexVector.getType().dyn_cast<mlir::VectorType>();
			if (!indexVectorType) {
				return nullptr;
			}

			mlir::Value index = builder.create<ConstantOp>(loc(call.getLocation()), builder.getIndexAttr(0));
			return builder.create<VectorGatherOp>(loc(call.getLocation()), mlir::VectorType::get(indexVectorType.getShape(), memRefType.getElementType()), memRef, index, indexVector);
		}

		mlir::Value mlirGenVectorHAdd(const ast::FunctionCall& call) {
			if (call.getArguments().size() != 1) {
				return nullptr;
			}

			auto vector = mlirGen(*call.getArguments().front());
			if (!vector) {
				return nullptr;
			}

			return builder.create<VectorHAddOp>(loc(call.getLocation()), builder.getI64Type(), vector);
		}

		mlir::Value mlirGenVectorLoad(const ast::FunctionCall& call) {
			if (call.getTemplateArguments().size() != 1 || call.getArguments().size() != 1) {
				return nullptr;
			}
			auto memref = mlirGen(*call.getArguments().front());
			if (!memref) {
				return nullptr;
			}
			auto length = std::get<ast::IntegerPtr>(call.getTemplateArgument(0))->getValue();

			mlir::Value index = builder.create<ConstantOp>(loc(call.getLocation()), builder.getIndexAttr(0));
			return builder.create<VectorLoadOp>(loc(call.getLocation()),
					mlir::VectorType::get(length, memref.getType().cast<mlir::MemRefType>().getElementType()), memref, index);
		}

		mlir::LogicalResult mlirGenVectorStore(const ast::FunctionCall& call) {
			if (call.getArguments().size() != 2) {
				return mlir::failure();
			}

			auto vector = mlirGen(*call.getArguments().front());
			if (!vector) {
				return mlir::failure();
			}

			auto memref = mlirGen(*call.getArguments().back());
			if (!memref) {
				return mlir::failure();
			}

			mlir::Value index = builder.create<ConstantOp>(loc(call.getLocation()), builder.getIndexAttr(0));
			builder.create<VectorStoreOp>(loc(call.getLocation()), vector, memref, index);
			return mlir::success();
		}

		mlir::Value mlirGenVectorSeq(const ast::FunctionCall& call) {
			if (call.getTemplateArguments().size() != 1 || call.getArguments().size() != 1) {
				return nullptr;
			}

			auto offset = mlirGen(*call.getArguments().front());
			if (!offset) {
				return nullptr;
			}

			auto length = std::get<ast::IntegerPtr>(call.getTemplateArgument(0))->getValue();

			return builder.create<VectorSequenceOp>(loc(call.getLocation()), mlir::VectorType::get(length, offset.getType()), offset);
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
				case i64:
					attribute = builder.getI64IntegerAttr(num.getAsSigned());
					break;
				case u64:
					attribute = builder.getI64IntegerAttr(num.getAsUnsigned());
					break;
				case i32:
					attribute = builder.getI32IntegerAttr(num.getAsSigned());
					break;
				case u32:
					attribute = builder.getI32IntegerAttr(num.getAsUnsigned());
					break;
				case i16:
					attribute = builder.getI16IntegerAttr(num.getAsSigned());
					break;
				case u16:
					attribute = builder.getI16IntegerAttr(num.getAsUnsigned());
					break;
				case i8:
					attribute = builder.getI8IntegerAttr(num.getAsSigned());
					break;
				case u8:
					attribute = builder.getI8IntegerAttr(num.getAsUnsigned());
					break;
				case usize:
					attribute = builder.getIndexAttr(num.getAsUnsigned());
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
		mlir::LogicalResult mlirGen(const ast::StatementPtrVec& blockAST) {
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
					if (func->getCallee() == "srand") {
						if (mlir::failed(mlirGenSrand(*func))) {
							return mlir::failure();
						}
						continue;
					}
					if (func->getCallee() == "vecCompressStore") {
						if (mlir::failed(mlirGenVectorCompressStore(*func))) {
							return mlir::failure();
						}
						continue;
					}
					if (func->getCallee() == "vecStore") {
						if (mlir::failed(mlirGenVectorStore(*func))) {
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

		mlir::Type getMlirType(const TypeType& type) {
			switch (type) {
				case i64:
				case u64:
					return builder.getI64Type();
				case i32:
				case u32:
					return builder.getI32Type();
				case i16:
				case u16:
					return builder.getIntegerType(16);
				case i8:
				case u8:
					return builder.getIntegerType(8);
				case usize:
					return builder.getIndexType();
				case f64:
					return builder.getF64Type();
				case f32:
					return builder.getF32Type();
				case boolean:
					return builder.getI1Type();

				case unknown:
				case number:
				case integer:
				case floatingPoint:
				case array:
				case void_:
				case vec:
				case mask:
				case range:
				case callable:
				case string:
					assert(false && "type not supported");
					return builder.getNoneType();
			}
		}

		mlir::Value getOrCreateGlobalString(mlir::Location loc, StringRef value) {
			return getOrCreateGlobalString(loc, "str_" + std::to_string(hash_value(value)), value);
		}

		/// Return a value representing an access into a global string with the given name, creating the string if
		/// necessary.
		mlir::Value getOrCreateGlobalString(mlir::Location loc, StringRef name, StringRef value) {
			// Create the global at the entry of the module.
			mlir::LLVM::GlobalOp global;
			if (!(global = theModule.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
				mlir::OpBuilder::InsertionGuard insertGuard(builder);
				builder.setInsertionPointToStart(theModule.getBody());
				auto type = mlir::LLVM::LLVMArrayType::get(mlir::IntegerType::get(builder.getContext(), 8), value.size());
				global = builder.create<mlir::LLVM::GlobalOp>(loc, type, true, mlir::LLVM::Linkage::Internal, name,
				                                        builder.getStringAttr(value));
			}

			// Get the pointer to the first character in the global string.
			mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);
			mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(loc, mlir::IntegerType::get(builder.getContext(), 64),
			                                              builder.getIntegerAttr(builder.getIndexType(), 0));
			return builder.create<mlir::LLVM::GEPOp>(loc,
			                                         mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(builder.getContext(), 8)), globalPtr,
			                                   ArrayRef<mlir::Value>({cst0, cst0}));
		}
	};

} // namespace

namespace tvl {

	// The public API for codegen.
	mlir::OwningModuleRef mlirGen(mlir::MLIRContext& context, ast::Module& moduleAST) {
		context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
		return MLIRGenImpl(context).mlirGen(moduleAST);
	}

} // namespace tvl
