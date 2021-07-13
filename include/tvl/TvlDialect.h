#ifndef TVL_DIALECT_H
#define TVL_DIALECT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/TypeSupport.h"

#include "tvl/TvlOpsDialect.h.inc"

namespace mlir {
	class MLIRContext;

	namespace tvl {
		struct StructTypeStorage;

		class StructType : public mlir::Type::TypeBase<StructType, mlir::Type, StructTypeStorage> {
		public:
			using Base::Base;

			static StructType get(MLIRContext *context, llvm::ArrayRef<mlir::Type> elementTypes);
			llvm::ArrayRef<mlir::Type> getElementTypes();
			size_t getNumElementTypes() { return getElementTypes().size(); }
		};
	}
}

#endif // TVL_DIALECT_H
