#include "tvl/TvlDialect.h"
#include "tvl/TvlOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::tvl;

/** ConstantOp **/

static mlir::ParseResult parseConstantOp(mlir::OpAsmParser& parser, mlir::OperationState& result) {
	mlir::IntegerAttr value;
	if (parser.parseOptionalAttrDict(result.attributes) || parser.parseAttribute(value, "value", result.attributes)) {
		return failure();
	}

	result.addTypes(value.getType());
	return success();
}

static void print(mlir::OpAsmPrinter& printer, ConstantOp op) {
	printer << ConstantOp::getOperationName() << " ";
	printer.printOptionalAttrDict(op->getAttrs(), {"value"});
	printer << op.value();
}

static mlir::LogicalResult verify(ConstantOp op) {
	if (op.getResult().getType().isa<mlir::NoneType>()) {
		return op.emitError("Constant must not return None.");
	}

	return mlir::success();
}

/** ForOp **/

ParseResult parseForOp(OpAsmParser& parser, OperationState& result) {
	auto& builder = parser.getBuilder();

	Type indexType = builder.getIndexType();
	OpAsmParser::OperandType inductionVariable, begin, end;
	if (parser.parseRegionArgument(inductionVariable) || parser.parseKeyword("in") || parser.parseOperand(begin) ||
			parser.resolveOperand(begin, indexType, result.operands) || parser.parseKeyword("to") ||
			parser.parseOperand(end) || parser.resolveOperand(end, indexType, result.operands)) {
		return failure();
	}

	SmallVector<OpAsmParser::OperandType, 4> regionArguments;
	regionArguments.push_back(inductionVariable);

	SmallVector<Type, 4> regionArgumentTypes;
	regionArgumentTypes.push_back(indexType);

	if (regionArguments.size() != regionArgumentTypes.size()) {
		return parser.emitError(parser.getNameLoc(),
				"number of loop-carried values does not equal number of defined values");
	}

	Region* body = result.addRegion();
	if (parser.parseRegion(*body, regionArguments, regionArgumentTypes)) {
		return failure();
	}

	ForOp::ensureTerminator(*body, builder, result.location);

	if (parser.parseOptionalAttrDict(result.attributes)) {
		return failure();
	}

	return success();
}

void print(OpAsmPrinter& printer, ForOp op) {
	printer << ForOp::getOperationName() << " " << op.getInductionVariable() << " in " << op.begin() << " to "
			<< op.end();

	printer.printRegion(op.region(), false, op.hasIterOperands());
	printer.printOptionalAttrDict(op->getAttrs());
}

/** GenericCallOp **/

void GenericCallOp::build(OpBuilder& builder, OperationState& state, StringRef callee, ArrayRef<Value> arguments) {
	// Generic call always returns an unranked Tensor initially.
	state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
	state.addOperands(arguments);
	state.addAttribute("callee", builder.getSymbolRefAttr(callee));
}

#define GET_OP_CLASSES

#include "tvl/TvlOps.cpp.inc"
