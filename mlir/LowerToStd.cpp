#include "tvl/TvlDialect.h"
#include "tvl/TvlOps.h"
#include "tvl/Passes.h"

#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
	template<typename BinaryOperator, typename BinaryLoweredOperation>
	class BinaryOpLowering : public ConversionPattern {
	public:
		explicit BinaryOpLowering(MLIRContext* context)
				: ConversionPattern{BinaryOperator::getOperationName(), 1, context} {}

		LogicalResult
		matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			rewriter.replaceOpWithNewOp<BinaryLoweredOperation>(op, operands);
			return success();
		}
	};

	using AddOpLowering = BinaryOpLowering<tvl::AddOp, AddIOp>;
	using SubOpLowering = BinaryOpLowering<tvl::SubOp, SubIOp>;
	using MulOpLowering = BinaryOpLowering<tvl::MulOp, MulIOp>;
	using DivOpLowering = BinaryOpLowering<tvl::DivOp, UnsignedDivIOp>;
	using RemOpLowering = BinaryOpLowering<tvl::RemOp, UnsignedRemIOp>;

	class ConstantOpLowering : public ConversionPattern {
	public:
		explicit ConstantOpLowering(MLIRContext* context)
				: ConversionPattern(tvl::ConstantOp::getOperationName(), 1, context) {}

		LogicalResult
		matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			auto constantOp = cast<tvl::ConstantOp>(op);
			rewriter.replaceOpWithNewOp<ConstantOp>(op, constantOp.value());
			return success();
		}
	};

	class LoadOpLowering : public ConversionPattern {
	public:
		explicit LoadOpLowering(MLIRContext* context)
				: ConversionPattern{tvl::LoadOp::getOperationName(), 1, context} {}

		LogicalResult
		matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			auto loadOp = cast<tvl::LoadOp>(op);
			auto location = loadOp.getLoc();
			auto memRef = loadOp.memRef();
			auto index = loadOp.index();

			Value dim = rewriter.create<memref::DimOp>(location, memRef, 0);
			auto pred = rewriter.create<CmpIOp>(location, CmpIPredicate::ult, index, dim);
			rewriter.create<AssertOp>(location, pred, rewriter.getStringAttr("Out of Bounds"));

			rewriter.replaceOpWithNewOp<memref::LoadOp>(op, memRef, index);
			return success();
		}
	};

	class ReturnOpLowering : public ConversionPattern {
	public:
		explicit ReturnOpLowering(MLIRContext* context)
				: ConversionPattern(tvl::ReturnOp::getOperationName(), 1, context) {}

		LogicalResult
		matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			auto returnOp = cast<tvl::ReturnOp>(op);
			if (returnOp.hasOperand()) {
				return failure();
			}

			rewriter.replaceOpWithNewOp<ReturnOp>(op);
			return success();
		}
	};

	class StoreOpLowering : public ConversionPattern {
	public:
		explicit StoreOpLowering(MLIRContext* context)
				: ConversionPattern{tvl::StoreOp::getOperationName(), 1, context} {}

		LogicalResult
		matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			auto storeOp = cast<tvl::StoreOp>(op);
			auto location = storeOp.getLoc();
			auto value = storeOp.value();
			auto memRef = storeOp.memRef();
			auto index = storeOp.index();

			Value dim = rewriter.create<memref::DimOp>(location, memRef, 0);
			auto pred = rewriter.create<CmpIOp>(location, CmpIPredicate::ult, index, dim);
			rewriter.create<AssertOp>(location, pred, rewriter.getStringAttr("Out of Bounds"));

			rewriter.replaceOpWithNewOp<memref::StoreOp>(op, value, memRef, index);
			return success();
		}
	};

	class VectorBroadcastOpLowering : public ConversionPattern {
	public:
		explicit VectorBroadcastOpLowering(MLIRContext* context)
				: ConversionPattern{tvl::VectorBroadcastOp::getOperationName(), 1, context} {}

		LogicalResult
		matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			auto broadcastOp = cast<tvl::VectorBroadcastOp>(op);

			rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, broadcastOp.vectorType(), broadcastOp.source());
			return success();
		}
	};

	class VectorExtractElementOpLowering : public ConversionPattern {
	public:
		explicit VectorExtractElementOpLowering(MLIRContext* context)
				: ConversionPattern{tvl::VectorExtractElementOp::getOperationName(), 1, context} {}

		LogicalResult
		matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			auto vectorExtractElementOp = cast<tvl::VectorExtractElementOp>(op);

			rewriter.replaceOpWithNewOp<vector::ExtractElementOp>(op, vectorExtractElementOp.elementType(),
					vectorExtractElementOp.vector(), vectorExtractElementOp.index());
			return success();
		}
	};

	class VectorHAddOpLowering : public ConversionPattern {
	public:
		explicit VectorHAddOpLowering(MLIRContext* context)
				: ConversionPattern{tvl::VectorHAddOp::getOperationName(), 1, context} {}

		LogicalResult
		matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			auto vectorHAddOp = cast<tvl::VectorHAddOp>(op);

			rewriter.replaceOpWithNewOp<vector::ReductionOp>(op, vectorHAddOp.resultType(), "add",
					vectorHAddOp.vector(), ValueRange({}));
			return success();
		}
	};

	class VectorLoadOpLowering : public ConversionPattern {
	public:
		explicit VectorLoadOpLowering(MLIRContext* context)
				: ConversionPattern{tvl::VectorLoadOp::getOperationName(), 1, context} {}

		LogicalResult
		matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			auto vectorLoadOp = cast<tvl::VectorLoadOp>(op);

			rewriter.replaceOpWithNewOp<vector::LoadOp>(op, vectorLoadOp.vectorType(), vectorLoadOp.base(),
					vectorLoadOp.indices());
			//rewriter.replaceOpWithNewOp<vector::ReductionOp>(op, vectorHAddOp.resultType(), "add",
			//		vectorHAddOp.vector(), ValueRange({}));
			return success();
		}
	};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// TvlToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
	struct TvlToStdLoweringPass : public PassWrapper<TvlToStdLoweringPass, OperationPass<ModuleOp>> {
		void getDependentDialects(DialectRegistry& registry) const final {
			registry.insert<StandardOpsDialect, scf::SCFDialect, vector::VectorDialect>();
		}

		void runOnOperation() final;
	};
} // end anonymous namespace

void TvlToStdLoweringPass::runOnOperation() {
	// The first thing to define is the conversion target. This will define the final target for this lowering. For this
	// lowering, we are only targeting the LLVM dialect.
	ConversionTarget target(getContext());
	target.addLegalDialect<memref::MemRefDialect, StandardOpsDialect, vector::VectorDialect>();
	target.addLegalOp<ModuleOp, FuncOp, tvl::PrintOp, tvl::RandOp, tvl::SRandOp/*, ModuleTerminatorOp*/>();

	// Now that the conversion target has been defined, we need to provide the patterns used for lowering. At this point
	// of the compilation process, we have a combination of `tvl`, `affine`, and `std` operations. Luckily, there are
	// already exists a set of patterns to transform `affine` and `std` dialects. These patterns lowering in multiple
	// stages, relying on transitive lowerings. Transitive lowering, or A->B->C lowering, is when multiple patterns must
	// be applied to fully transform an illegal operation into a set of legal ones.
	RewritePatternSet patterns(&getContext());
	//populateAffineToStdConversionPatterns(patterns, &getContext());
	populateLoopToStdConversionPatterns(patterns);

	// The only remaining operation to lower from the `tvl` dialect, is the PrintOp.
	patterns.insert<AddOpLowering, ConstantOpLowering, DivOpLowering, LoadOpLowering, MulOpLowering, RemOpLowering,
			ReturnOpLowering, StoreOpLowering, SubOpLowering, VectorBroadcastOpLowering, VectorExtractElementOpLowering,
			VectorHAddOpLowering, VectorLoadOpLowering>(&getContext());

	// We want to completely lower to LLVM, so we use a `FullConversion`. This ensures that only legal operations will
	// remain after the conversion.
	auto module = getOperation();
	if (failed(applyFullConversion(module, target, std::move(patterns)))) {
		signalPassFailure();
	}
}

/// Create a pass for lowering operations the remaining `TVL` operations, as well as `SCF` and `Std`, to the LLVM
/// dialect for codegen.
std::unique_ptr<Pass> tvl::createLowerToStdPass() {
	return std::make_unique<TvlToStdLoweringPass>();
}
