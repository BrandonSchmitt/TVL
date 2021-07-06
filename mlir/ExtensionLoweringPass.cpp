#include "tvl/Passes.h"
#include "tvl/TvlDialect.h"
#include "tvl/TvlOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

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
			auto end = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(vectorSequenceOp.vectorType().getShape().front()));
			auto step = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));

			rewriter.create<scf::ForOp>(loc, begin, end, step, llvm::None,
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

	class MaskPopulationCountOpLowering : public ConversionPattern {
	public:
		explicit MaskPopulationCountOpLowering(MLIRContext* context) : ConversionPattern{tvl::MaskCountTrueOp::getOperationName(), 1, context} {}

		LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			if (succeeded(rewriteFixedBitLength(op, rewriter, 64))) {
				return success();
			}
			if (succeeded(rewriteFixedBitLength(op, rewriter, 32))) {
				return success();
			}
			if (succeeded(rewriteFixedBitLength(op, rewriter, 16))) {
				return success();
			}
			if (succeeded(rewriteFixedBitLength(op, rewriter, 8))) {
				return success();
			}

			auto maskCountTrueOp = cast<tvl::MaskCountTrueOp>(op);
			auto loc = maskCountTrueOp->getLoc();
			auto mask = maskCountTrueOp.mask();
			auto maskLength = maskCountTrueOp.maskLength();

			auto begin = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0));
			auto end = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(maskLength));
			auto step = rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1));

			Value const0u64 = rewriter.create<ConstantOp>(loc, rewriter.getI64IntegerAttr(0));

			auto loop = rewriter.create<scf::ForOp>(loc, begin, end, step, ValueRange(const0u64),
					[&](OpBuilder& builder, Location loc, Value inductionVar, ValueRange valueRange) {
						Value castInductionVar = builder.create<IndexCastOp>(loc, inductionVar, builder.getI64Type());
						Value bit = builder.create<vector::ExtractElementOp>(loc, mask, castInductionVar);
						Value castBit = builder.create<ZeroExtendIOp>(loc, bit, builder.getI64Type());
						Value value = builder.create<AddIOp>(loc, valueRange.front(), castBit);
						builder.create<scf::YieldOp>(loc, ValueRange(value));
					});

			rewriter.replaceOpWithNewOp<IndexCastOp>(op, loop.getResult(0), rewriter.getIndexType());

			return success();
		}

	protected:
		LogicalResult rewriteFixedBitLength(Operation* op, ConversionPatternRewriter& rewriter, int64_t bitLength) const {
			auto maskCountTrueOp = cast<tvl::MaskCountTrueOp>(op);
			auto maskLength = maskCountTrueOp.maskLength();
			if (maskLength % bitLength == 0 && maskLength <= ((1 << bitLength) - 1)) {
				auto loc = maskCountTrueOp->getLoc();
				Type type = rewriter.getIntegerType(bitLength);
				Value bitCast = rewriter.create<vector::BitCastOp>(loc, VectorType::get(maskLength / bitLength, type), maskCountTrueOp.mask());
				Value vectorPopCount = rewriter.create<LLVM::CtPopOp>(loc, bitCast);
				Value totalPopCount = rewriter.create<vector::ReductionOp>(loc, type, "add", vectorPopCount, ValueRange({}));
				rewriter.replaceOpWithNewOp<IndexCastOp>(op, totalPopCount, rewriter.getIndexType());
				return success();
			}

			return failure();
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
	target.addLegalDialect<tvl::TvlDialect, memref::MemRefDialect, scf::SCFDialect, StandardOpsDialect, vector::VectorDialect, LLVM::LLVMDialect>();
	target.addIllegalOp<tvl::VectorSequenceOp, tvl::MaskCountTrueOp>();

	// The only remaining operation to lower from the `tvl` dialect, is the PrintOp.
	RewritePatternSet patterns(&getContext());
	patterns.insert<VectorSequenceOpLowering, MaskPopulationCountOpLowering>(&getContext());

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
	target.addLegalDialect<tvl::TvlDialect, memref::MemRefDialect, scf::SCFDialect, StandardOpsDialect, vector::VectorDialect, LLVM::LLVMDialect>();
	target.addIllegalOp<tvl::VectorSequenceOp, tvl::MaskCountTrueOp>();

	// The only remaining operation to lower from the `tvl` dialect, is the PrintOp.
	RewritePatternSet patterns(&getContext());
	patterns.insert<VectorSequenceOpLowering, MaskPopulationCountOpLowering>(&getContext());

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
