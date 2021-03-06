#ifndef TVL_DIALECT_PASSES_H
#define TVL_DIALECT_PASSES_H

#include <memory>

namespace mlir {
	class Pass;

	namespace tvl {
		std::unique_ptr<mlir::Pass> createLowerToSCFPass();
		std::unique_ptr<mlir::Pass> createLowerToStdPass();
		std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
		std::unique_ptr<mlir::Pass> createAVX512LoweringPass();
		std::unique_ptr<mlir::Pass> createNoExtensionLoweringPass();
	} // namespace tvl
} // namespace mlir

#endif //TVL_DIALECT_PASSES_H
