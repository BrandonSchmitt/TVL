#ifndef TVL_OPS
#define TVL_OPS

include "TvlDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AddOp : TvlBinaryOp<"add", [Commutative]> {
	let summary = "element-wise addition operation";
	let description = [{
		The "add" operation performs element-wise addition between two tensors.
		The shapes of the tensor operands are expected to match.
	}];
}
def DivOp : TvlBinaryOp<"div"> {}
def MulOp : TvlBinaryOp<"mul", [Commutative]> {}
def RemOp : TvlBinaryOp<"rem"> {}
def SubOp : TvlBinaryOp<"sub"> {}

def AndOp : TvlBinaryOp<"and", [Commutative]> {}
def OrOp : TvlBinaryOp<"or", [Commutative]> {}
def XOrOp : TvlBinaryOp<"xor", [Commutative]> {}

def MinOp : TvlBinaryOp<"min", [Commutative]> {}
def MaxOp : TvlBinaryOp<"max", [Commutative]> {}

def ShiftLeftOp : TvlBinaryOp<"shift_left"> {}
def ShiftRightUnsignedOp : TvlBinaryOp<"shift_right_unsigned"> {}
def ShiftRightSignedOp : TvlBinaryOp<"shift_right_signed"> {}

def EqOp : TvlBinaryCmpOp<"eq", [Commutative]> {}
def NeOp : TvlBinaryCmpOp<"ne", [Commutative]> {}
def SgeOp : TvlBinaryCmpOp<"sge"> {}
def SgtOp : TvlBinaryCmpOp<"sgt"> {}
def SleOp : TvlBinaryCmpOp<"sle"> {}
def SltOp : TvlBinaryCmpOp<"slt"> {}
def UgeOp : TvlBinaryCmpOp<"uge"> {}
def UgtOp : TvlBinaryCmpOp<"ugt"> {}
def UleOp : TvlBinaryCmpOp<"ule"> {}
def UltOp : TvlBinaryCmpOp<"ult"> {}

class MaskCountOp<string mnemonic> : TvlOp<mnemonic, [NoSideEffect]> {
	let arguments = (ins VectorOfRankAndType<[1], [I1]>:$mask);
	let results = (outs Index:$count);

	let builders = [
		OpBuilder<(ins "Value":$mask), [{ build($_builder, $_state, $_builder.getIndexType(), mask); }]>
	];

	let extraClassDeclaration = [{
		auto maskLength() { return mask().getType().cast<VectorType>().getShape().front(); }
	}];

	let assemblyFormat = "$mask attr-dict `:` type($mask)";
}

def MaskCountTrueOp : MaskCountOp<"mask_count_true"> {}
def MaskCountFalseOp : MaskCountOp<"mask_count_false"> {}

def ConstantOp : TvlOp<"constant", [NoSideEffect]> {
	let summary = "constant";
	let description = [{
		Constant operation turns a literal into an SSA value. The data is attached to the operation as an attribute.
		For example:

		```mlir
			%0 = tvl.constant 42 : i64
			%1 = tvl.constant 11 : index
		```
	}];

	let arguments = (ins AnyAttr:$value);
	let results = (outs AnyType);

	let printer = [{ return ::print(p, *this); }];
	let parser = [{ return ::parse$cppClass(parser, result); }];

	let builders = [
		OpBuilder<(ins "Attribute":$value), [{ build($_builder, $_state, value.getType(), value); }]>
	];

	let verifier = [{ return ::verify(*this); }];
}

def ForOp : TvlOp<"for", [SingleBlockImplicitTerminator<"tvl::YieldOp">]> {
	let summary = "iterator loop";
	let description = [{
		Iterates through a collection or range executing the loop's body for each iteration.

		```mlir
		%0 = tvl.constant 0 : index
		%1 = tvl.constant 10 : index
		tvl.for %i = %0 to %1 {
			%2 = tvl.constant 42 : i64
			tvl.print %2 : i64
		}
		```
	}];

	let arguments = (ins Index:$begin, Index:$end);
	let results = (outs Variadic<AnyType>:$reuslts);
	let regions = (region SizedRegion<1>:$region);

	let printer = [{ return ::print(p, *this); }];
	let parser = [{ return ::parse$cppClass(parser, result); }];
	//let verifier = [{ return ::verify(*this); }];

	let extraClassDeclaration = [{
		Value getInductionVariable() { return getBody()->getArgument(0); }

		unsigned int getNumControlOperands() { return 2; /* begin, end */ }

		unsigned int getNumIterOperands() { return getOperation()->getNumOperands() - getNumControlOperands(); }
		bool hasIterOperands() { return getNumIterOperands() > 0; }
		Operation::operand_range getIterOperands() { return getOperands().drop_front(getNumControlOperands()); }
	}];
}

def GenericCallOp : TvlOp<"generic_call"> {
	let summary = "generic call operation";
	let description = [{
		Generic calls represent calls to a user defined function that needs to be specialized for the shape of its
		arguments. The callee name is attached as a symbol reference via an attribute. The arguments list must match the
		arguments expected by the callee. For example:

		```mlir
			%4 = tvl.generic_call @my_func(%1, %3) : (i64, i64) -> i64
		```

		This is only valid if a function named "my_func" exists and takes two arguments.
	}];

	// The generic call operation takes a symbol reference attribute as the callee, and inputs for the call.
	let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<I64>:$inputs);

	// The generic call operation returns a single value of integer.
	let results = (outs I64);

	// Specialize assembly printing and parsing using a declarative format.
	//let assemblyFormat = [{
	//	$callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
	//}];

	// Add custom build methods for the generic call operation.
	let builders = [
		OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
	];
}

def LoadOp : TvlOp<"load", [
			MemRefsNormalizable,
			TypesMatchWith<"result type matches element type of 'memRef'", "memRef", "result", "$_self.cast<MemRefType>().getElementType()">
		]> {
	let summary = "Load an element from a mem-ref";
	let description = [{}];

	let arguments = (ins Arg<AnyMemRef, "the reference to load from", [MemRead]>:$memRef, Index:$index);
	let results = (outs AnyType:$result);

	let builders = [
		OpBuilder<(ins "Value":$memRef, "Value":$index), [{
			auto memRefType = memRef.getType().cast<MemRefType>();
			$_state.addOperands(memRef);
			$_state.addOperands(index);
			$_state.types.push_back(memRefType.getElementType());
		}]>
	];

	let extraClassDeclaration = [{
		Value getMemRef() { return getOperand(0); }
		MemRefType memRefType() { return memRef().getType().cast<MemRefType>(); }

		Value getIndex() { return getOperand(1); }
	}];

	//let hasCanonicalizer = 1;
	//let hasFolder = 1;

	let assemblyFormat = "$memRef `[` $index `]` attr-dict `:` type($memRef)";
}

def PrintOp : TvlOp<"print"> {
	let summary = "print operation";
	let description = [{
		The "print" builtin operation prints a given input integer, and produces no results.
	}];

	let arguments = (ins LLVM_AnyPointer:$formatPointer, AnyTypeOf<[I8, I16, I32, I64, Index]>:$input);

	let assemblyFormat = "$formatPointer `,` $input attr-dict `:` type($formatPointer) `,` type($input)";
}

def RandOp : TvlOp<"rand"> {
	let summary = "Generates a random value";

	let results = (outs I64:$value);

	let assemblyFormat = "attr-dict `:` type($value)";
}

def ReturnOp : TvlOp<"return", [NoSideEffect, HasParent<"FuncOp">, Terminator]> {
	let summary = "return operation";
	let description = [{
		The "return" operation represents a return operation within a function.
		The operation takes an optional tensor operand and produces no results.
		The operand type must match the signature of the function that contains the operation. For example:

		```mlir
		func @foo() -> i64 {
			...
			tvl.return %0 : i64
		}
		```
	}];

	// The return operation takes an optional input operand to return. This value must match the return type of the
	// enclosing function.
	let arguments = (ins Variadic<I64>:$input);

	// The return operation only emits the input in the format if it is present.
	let assemblyFormat = "($input^ `:` type($input))? attr-dict";

	let builders = [
		OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>	// ReturnOp with no operand
	];

	// Provide extra utility definitions on the c++ operation class definition.
	let extraClassDeclaration = [{
		bool hasOperand() { return getNumOperands() != 0; }
	}];

	// Invoke a static verify method to verify this return operation.
	//let verifier = [{ return ::verify(*this); }];
}

def SRandOp : TvlOp<"srand"> {
	let summary = "Seeds the random number generator";

	let arguments = (ins I32:$seed);

	let assemblyFormat = "$seed attr-dict `:` type($seed)";
}

def StoreOp : TvlOp<"store", [
			MemRefsNormalizable,
			TypesMatchWith<"type of 'value' matches element type of 'memRef'", "memRef", "value", "$_self.cast<MemRefType>().getElementType()">,
		]> {
	let summary = "stores an element into a mem-ref";
	let description = [{}];

	let arguments = (ins AnyType:$value, Arg<AnyMemRef, "the reference to store the value in", [MemWrite]>:$memRef, Index:$index);

	let builders = [
		OpBuilder<(ins "Value":$value, "Value":$memRef), [{
			$_state.addOperands(value);
			$_state.addOperands(memRef);
		}]>
	];

	let extraClassDeclaration = [{
		MemRefType memRefType() { return memRef().getType().cast<MemRefType>(); }
	}];

	//let hasFolder = 1;

	let assemblyFormat = [{
		$value `,` $memRef `[` $index `]` attr-dict `:` type($memRef)
	}];
}

def VectorBroadcastOp : TvlOp<"vector_broadcast", [NoSideEffect]> {
	let summary = "Broadcast a value to a vector";
	let description = [{
		```mlir
		%0 = tvl.constant 1 : i64
		%1 = tvl.vector_broadcast %0 : vector<4xi64>
		```
	}];

	let arguments = (ins AnyType:$source);
	let results = (outs AnyVector:$vector);

	let extraClassDeclaration = [{
		Type sourceType() { return source().getType(); }
		VectorType vectorType() { return vector().getType().cast<VectorType>(); }
	}];

	let assemblyFormat = "$source attr-dict `:` type($source) `to` type($vector)";
}

def VectorCompressStoreOp : TvlOp<"vector_compress_store", []> {
	let arguments = (ins VectorOfRank<[1]>:$valueToStore, VectorOfRankAndType<[1], [I1]>:$mask, Arg<AnyMemRef, "", [MemWrite]>:$base, Variadic<Index>:$indices);

	let extraClassDeclaration = [{
		MemRefType getMemRefType() { return base().getType().cast<MemRefType>(); }
		VectorType getMaskVectorType() { return mask().getType().cast<VectorType>(); }
		VectorType getVectorType() { return valueToStore().getType().cast<VectorType>(); }
	}];
	let assemblyFormat = "$valueToStore `,` $mask `,` $base `[` $indices `]` attr-dict `:` "
			"type($valueToStore) `,` type($mask) `,` type($base)";
}

def VectorExtractElementOp : TvlOp<"vector_extract_element", [
		NoSideEffect,
		//PredOpTrait<"source operand and result have same element type", TCresVTEtIsSameAsOpBase<0, 0>>
		TypesMatchWith<"result type matches element type of vector operand", "vector", "element", "$_self.cast<ShapedType>().getElementType()">
		]> {
	let summary = "Extracts one element from an 1-D vector";

	let arguments = (ins AnyVector:$vector, AnySignlessInteger:$index);
	let results = (outs AnyType:$element);

	let extraClassDeclaration = [{
		Type elementType() { return element().getType(); }
		VectorType vectorType() { return vector().getType().cast<VectorType>(); }
	}];

	let assemblyFormat = "$vector `[` $index `:` type($index) `]` attr-dict `:` type($vector)";
}

def VectorGatherOp : TvlOp<"vector_gather"> {
	let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$base, Variadic<Index>:$indices, VectorOfRankAndType<[1], [AnyInteger]>:$indexVector);
	let results = (outs VectorOfRank<[1]>:$result);

	let extraClassDeclaration = [{
		MemRefType memRefType() { return base().getType().cast<MemRefType>(); }
		VectorType indexVectorType() { return indexVector().getType().cast<VectorType>(); }
		VectorType resultType() { return result().getType().cast<VectorType>(); }
	}];

	let assemblyFormat = "$base `[` $indices `]` `[` $indexVector `]` attr-dict `:` type($base) `,` type($indexVector) `into` type($result)";
}

def VectorHAddOp : TvlOp<"vector_hadd", [
		NoSideEffect,
		PredOpTrait<"source operand and result have same element type", TCresVTEtIsSameAsOpBase<0, 0>>
		]> {
	let summary = "\"Horizontally\" adds the elements of a vector into a scalar";

	let arguments = (ins AnyVector:$vector);
	let results = (outs AnyType:$result);

	let extraClassDeclaration = [{
		Type resultType() { return result().getType(); }
		VectorType vectorType() { return vector().getType().cast<VectorType>(); }
	}];

	let assemblyFormat = "$vector attr-dict `:` type($vector) `to` type($result)";
}

def VectorLoadOp : TvlOp<"vector_load", [NoSideEffect]> {
	let summary = "Loads data from memory into a vector";

	let arguments = (ins Arg<AnyMemRef, "the reference to load from", [MemRead]>:$base, Variadic<Index>:$indices);
	let results = (outs AnyVector:$result);

	let extraClassDeclaration = [{
		MemRefType memRefType() { return base().getType().cast<MemRefType>(); }
		VectorType vectorType() { return result().getType().cast<VectorType>(); }
	}];

	let assemblyFormat = "$base `[` $indices `]` attr-dict `:` type($base) `,` type($result)";
}

def VectorSequenceOp : TvlOp<"vector_sequence", [NoSideEffect]> {
	let summary = "Initializes a vector with a sequence of integers";

	let arguments = (ins SignlessIntegerLike:$offset);
	let results = (outs AnyVector:$result);

	let extraClassDeclaration = [{
		VectorType vectorType() { return result().getType().cast<VectorType>(); }
	}];

	let assemblyFormat = "$offset attr-dict `:` type($offset) `in` type($result)";
}

def VectorStoreOp : TvlOp<"vector_store", []> {
	let arguments = (ins AnyVector:$vector, Arg<AnyMemRef, "the reference to store to", [MemWrite]>:$base, Variadic<Index>:$indices);

	let extraClassDeclaration = [{
		MemRefType memRefType() { return base().getType().cast<MemRefType>(); }
		VectorType vectorType() { return vector().getType().cast<VectorType>(); }
	}];

	let assemblyFormat = "$vector `,` $base `[` $indices `]` attr-dict `:` type($base) `,` type($vector)";
}

def YieldOp : TvlOp<"yield", [NoSideEffect, /*ReturnLike,*/ Terminator, ParentOneOf<["ForOp"]>]> {
	let summary = "Loop yield and terminator";
	let description = "";

	let arguments = (ins Variadic<AnyType>:$results);
	let builders = [OpBuilder<(ins), [{ /* do nothing */ }]>];

	let assemblyFormat = [{ attr-dict ($results^ `:` type($results))? }];
}

#endif // TVL_OPS
