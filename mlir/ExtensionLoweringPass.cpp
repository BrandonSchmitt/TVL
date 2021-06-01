#include "tvl/Passes.h"
#include "tvl/TvlDialect.h"
#include "tvl/TvlOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
	class VectorSequenceOpLowering : public ConversionPattern {
	public:
		explicit VectorSequenceOpLowering(MLIRContext* context) : ConversionPattern{tvl::VectorSequenceOp::getOperationName(), 1, context} {}

		LogicalResult
		matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			auto vectorSequenceOp = cast<tvl::VectorSequenceOp>(op);
			auto loc = op->getLoc();

			auto memRefType = mlir::MemRefType::get(vectorSequenceOp.vectorType().getShape(), vectorSequenceOp.vectorType().getElementType());
			auto memRef = rewriter.create<mlir::memref::AllocaOp>(loc, memRefType);

			auto begin = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
			auto step = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));

			rewriter.create<scf::ForOp>(loc, begin, vectorSequenceOp.length(), step, llvm::None,
					[&](OpBuilder& builder, Location loc, Value inductionVar, ValueRange) {
						auto castInductionVar = builder.create<IndexCastOp>(loc, inductionVar, vectorSequenceOp.offset().getType());
						auto value = builder.create<AddIOp>(loc, vectorSequenceOp.offset(), castInductionVar);
						builder.create<memref::StoreOp>(loc, value, memRef, inductionVar);
						builder.create<scf::YieldOp>(loc);
					});

			mlir::Value index = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
			rewriter.replaceOpWithNewOp<vector::LoadOp>(op, vectorSequenceOp.vectorType(), memRef, index);

			//rewriter.eraseBlock(scfForOp.getBody());
			//rewriter.inlineRegionBefore(forOp.region(), scfForOp.region(), scfForOp.region().end());
			//rewriter.replaceOp(op, scfForOp.results());

			return success();
		}
	};
}

namespace {
	class NoExtensionLoweringPass : public PassWrapper<NoExtensionLoweringPass, OperationPass<ModuleOp>> {
	public:
		void getDependentDialects(DialectRegistry& registry) const final {
			registry.insert<scf::SCFDialect, StandardOpsDialect, vector::VectorDialect>();
		}

		void runOnOperation() final;
	};
}

void NoExtensionLoweringPass::runOnOperation() {
	// The first thing to define is the conversion target. This will define the final target for this lowering. For this
	// lowering, we are only targeting the LLVM dialect.
	ConversionTarget target(getContext());
	target.addLegalOp<ModuleOp, /*ModuleTerminatorOp,*/ FuncOp, ConstantOp>();
	target.addLegalDialect<tvl::TvlDialect, memref::MemRefDialect, scf::SCFDialect, StandardOpsDialect, vector::VectorDialect>();
	target.addIllegalOp<tvl::VectorSequenceOp>();

	// The only remaining operation to lower from the `tvl` dialect, is the PrintOp.
	RewritePatternSet patterns(&getContext());
	patterns.insert<VectorSequenceOpLowering>(&getContext());

	// We want to completely lower to LLVM, so we use a `FullConversion`. This ensures that only legal operations will
	// remain after the conversion.
	auto module = getOperation();
	if (failed(applyFullConversion(module, target, std::move(patterns)))) {
		signalPassFailure();
	}
}

namespace {
	class AVX512LoweringPass : public PassWrapper<AVX512LoweringPass, OperationPass<ModuleOp>> {
	public:
		void getDependentDialects(DialectRegistry& registry) const final {
			registry.insert<scf::SCFDialect, StandardOpsDialect, vector::VectorDialect>();
		}

		void runOnOperation() final;
	};
}

void AVX512LoweringPass::runOnOperation() {
	// The first thing to define is the conversion target. This will define the final target for this lowering. For this
	// lowering, we are only targeting the LLVM dialect.
	ConversionTarget target(getContext());
	target.addLegalOp<ModuleOp, /*ModuleTerminatorOp,*/ FuncOp, ConstantOp>();
	target.addLegalDialect<tvl::TvlDialect, memref::MemRefDialect, scf::SCFDialect, StandardOpsDialect, vector::VectorDialect>();
	target.addIllegalOp<tvl::VectorSequenceOp>();

	// The only remaining operation to lower from the `tvl` dialect, is the PrintOp.
	RewritePatternSet patterns(&getContext());
	patterns.insert<VectorSequenceOpLowering>(&getContext());

	// We want to completely lower to LLVM, so we use a `FullConversion`. This ensures that only legal operations will
	// remain after the conversion.
	auto module = getOperation();
	if (failed(applyFullConversion(module, target, std::move(patterns)))) {
		signalPassFailure();
	}
}

/// Create a pass for lowering operations the remaining `TVL` operations, as well as `Affine` and `Std`, to the LLVM
/// dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::tvl::createNoExtensionLoweringPass() {
	return std::make_unique<NoExtensionLoweringPass>();
}

std::unique_ptr<mlir::Pass> mlir::tvl::createAVX512LoweringPass() {
	return std::make_unique<AVX512LoweringPass>();
}
