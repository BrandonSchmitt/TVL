#ifndef TVL_DIALECT_MLIRGEN_H
#define TVL_DIALECT_MLIRGEN_H

#include <memory>

namespace mlir {
	class MLIRContext;

	class OwningModuleRef;
} // namespace mlir

namespace tvl {
	namespace ast {
		class Module;
	}	// namespace tvl

	mlir::OwningModuleRef mlirGen(mlir::MLIRContext& context, ast::Module& moduleAST);
} // namespace tvl

#endif //TVL_DIALECT_MLIRGEN_H
