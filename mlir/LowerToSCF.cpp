#include "tvl/Passes.h"
#include "tvl/TvlDialect.h"
#include "tvl/TvlOps.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"


// Todo: maybe replace
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

namespace {
	class ForOpLowering : public ConversionPattern {
	public:
		explicit ForOpLowering(MLIRContext* context) : ConversionPattern{tvl::ForOp::getOperationName(), 1, context} {}

		LogicalResult
		matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			auto forOp = cast<tvl::ForOp>(op);

			auto loc = op->getLoc();
			auto step = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));

			auto scfForOp = rewriter.create<scf::ForOp>(loc, forOp.begin(), forOp.end(), step);
			rewriter.eraseBlock(scfForOp.getBody());
			rewriter.inlineRegionBefore(forOp.region(), scfForOp.region(), scfForOp.region().end());
			rewriter.replaceOp(op, scfForOp.results());

			return success();
		}
	};

	class YieldOpLowering : public ConversionPattern {
	public:
		explicit YieldOpLowering(MLIRContext* context) : ConversionPattern{tvl::YieldOp::getOperationName(), 1, context} {}

		LogicalResult
		matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			auto yieldOp = cast<tvl::YieldOp>(op);
			rewriter.replaceOpWithNewOp<scf::YieldOp>(op, yieldOp.results());
			return success();
		}
	};
}

namespace {
class TvlToSCFLoweringPass : public PassWrapper<TvlToSCFLoweringPass, OperationPass<ModuleOp>> {
	public:
		void getDependentDialects(DialectRegistry& registry) const final {
			registry.insert<scf::SCFDialect, StandardOpsDialect>();
		}

		void runOnOperation() final;
	};
}

void TvlToSCFLoweringPass::runOnOperation() {
	// The first thing to define is the conversion target. This will define the final target for this lowering. For this
	// lowering, we are only targeting the LLVM dialect.
	ConversionTarget target(getContext());
	target.addLegalOp<ModuleOp, /*ModuleTerminatorOp,*/ FuncOp, ConstantOp, LLVM::GlobalOp, LLVM::AddressOfOp, LLVM::ConstantOp, LLVM::GEPOp>();
	target.addLegalDialect<tvl::TvlDialect, scf::SCFDialect, memref::MemRefDialect>();
	target.addIllegalOp<tvl::ForOp, tvl::YieldOp>();

	// The only remaining operation to lower from the `tvl` dialect, is the PrintOp.
	RewritePatternSet patterns(&getContext());
	patterns.insert<ForOpLowering, YieldOpLowering>(&getContext());

	// We want to completely lower to LLVM, so we use a `FullConversion`. This ensures that only legal operations will
	// remain after the conversion.
	auto module = getOperation();
	if (failed(applyFullConversion(module, target, std::move(patterns)))) {
		signalPassFailure();
	}
}

/// Create a pass for lowering operations the remaining `TVL` operations, as well as `Affine` and `Std`, to the LLVM
/// dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::tvl::createLowerToSCFPass() {
	return std::make_unique<TvlToSCFLoweringPass>();
}
