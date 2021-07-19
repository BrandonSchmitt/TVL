#include "tvl/TvlDialect.h"
#include "tvl/TvlOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"

using namespace mlir;
using namespace mlir::tvl;

#include "tvl/TvlOpsDialect.cpp.inc"

namespace mlir::tvl {
	struct StructTypeStorage : public mlir::TypeStorage {
		using KeyTy = llvm::ArrayRef<mlir::Type>;

		StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
				: elementTypes(elementTypes) {}

		bool operator==(const KeyTy& key) const { return key == elementTypes; }

		/// Define a construction method for creating a new instance of this storage.
		/// This method takes an instance of a storage allocator, and an instance of a
		/// `KeyTy`. The given allocator must be used for *all* necessary dynamic
		/// allocations used to create the type storage and its internal.
		static StructTypeStorage* construct(mlir::TypeStorageAllocator& allocator, KeyTy const& key) {
			// Copy the elements from the provided `KeyTy` into the allocator.
			llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

			// Allocate the storage instance and construct it.
			return new(allocator.allocate<StructTypeStorage>()) StructTypeStorage(elementTypes);
		}

		llvm::ArrayRef<mlir::Type> elementTypes;
	};
}


StructType StructType::get(MLIRContext* context, llvm::ArrayRef<mlir::Type> elementTypes) {
	assert(!elementTypes.empty() && "expected at least 1 element type");

	// Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
	// of this type. The first parameter is the context to unique in. The
	// parameters after the context are forwarded to the storage instance.
	return Base::get(context, elementTypes);
}

llvm::ArrayRef<mlir::Type> StructType::getElementTypes() {
	return getImpl()->elementTypes;
}


/// Parse an instance of a type registered to the toy dialect.
mlir::Type TvlDialect::parseType(mlir::DialectAsmParser& parser) const {
	// Parse a struct type in the following form:
	//   struct-type ::= `struct` `<` type (`,` type)* `>`

	// Parse: `struct` `<`
	if (parser.parseKeyword("struct") || parser.parseLess()) {
		return Type();
	}

	// Parse the element types of the struct.
	SmallVector<mlir::Type, 1> elementTypes;
	do {
		// Parse the current element type.
		//auto typeLoc = parser.getCurrentLocation();
		mlir::Type elementType;
		if (parser.parseType(elementType)) {
			return nullptr;
		}

		// Check that the type is either a TensorType or another StructType.
		/*if (!elementType.isa<mlir::TensorType, StructType>()) {
			parser.emitError(typeLoc, "element type for a struct must either "
									  "be a TensorType or a StructType, got: ")
					<< elementType;
			return Type();
		}*/
		elementTypes.push_back(elementType);

		// Parse the optional: `,`
	}
	while (succeeded(parser.parseOptionalComma()));

	// Parse: `>`
	if (parser.parseGreater()) {
		return Type();
	}

	mlir::MLIRContext* ctx = elementTypes.front().getContext();
	return StructType::get(ctx, elementTypes);
}

/// Print an instance of a type registered to the toy dialect.
void TvlDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const {
	// Currently the only toy type is a struct type.
	auto structType = type.cast<StructType>();

	// Print the struct type according to the parser format.
	printer << "struct<";
	llvm::interleaveComma(structType.getElementTypes(), printer);
	printer << '>';
}

//===----------------------------------------------------------------------===//
// TVL dialect.
//===----------------------------------------------------------------------===//

void TvlDialect::initialize() {
	addOperations<
#define GET_OP_LIST

#include "tvl/TvlOps.cpp.inc"

	>();
	addTypes<StructType>();
}
