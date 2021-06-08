#ifndef TVL_DIALECT
#define TVL_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TvlDialect : Dialect {
    let name = "tvl";
    let summary = "A TVL MLIR dialect.";
    let cppNamespace = "::mlir::tvl";
}

class TvlOp<string mnemonic, list<OpTrait> traits = []> : Op<TvlDialect, mnemonic, traits>;

// Base class for standard binary arithmetic operations.
class TvlBinaryOp<string mnemonic, list<OpTrait> traits = []> : TvlOp<mnemonic, traits # [
		NoSideEffect,
		SameOperandsAndResultType,
	] # ElementwiseMappable.traits> {
	let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
	let results = (outs SignlessIntegerLike:$result);

	let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// Base class for standard binary arithmetic operations.
class TvlBinaryCmpOp<string mnemonic, list<OpTrait> traits = []> : TvlOp<mnemonic, traits # [
		NoSideEffect,
		SameTypeOperands,
		TypesMatchWith<"result type has i1 element type and same shape as operands", "lhs", "result", "getI1SameShape($_self)">
	] # ElementwiseMappable.traits> {
	let arguments = (ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs);
	let results = (outs SignlessIntegerLike:$result);

	let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
}

#endif // TVL_DIALECT
