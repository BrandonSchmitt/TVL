#include "tvl/TvlDialect.h"
#include "tvl/TvlOps.h"
#include "tvl/Passes.h"

#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

	class PrintOpLowering : public ConversionPattern {
	public:
		explicit PrintOpLowering(MLIRContext* context)
				: ConversionPattern(tvl::PrintOp::getOperationName(), 1, context) {}

		LogicalResult
		matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			auto loc = op->getLoc();

			ModuleOp parentModule = op->getParentOfType<ModuleOp>();

			// Get a symbol reference to the printf function, inserting it if necessary.
			auto printfRef = getOrInsertPrintf(rewriter, parentModule);
			Value formatSpecifierCst = getOrCreateGlobalString(loc, rewriter, "frmt_spec", StringRef("%lu\0", 4),
					parentModule);
			static_assert(sizeof(long int) == 8, "%lu is the wrong identifier for printf");
			Value newLineCst = getOrCreateGlobalString(loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

			auto printOp = cast<tvl::PrintOp>(op);
			rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
					ArrayRef<Value>({formatSpecifierCst, printOp.input()}));
			rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32), newLineCst);

			// Notify the rewriter that this operation has been removed.
			rewriter.eraseOp(op);
			return success();
		}

	private:
		/// Return a symbol reference to the printf function, inserting it into the module if necessary.
		static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter& rewriter, ModuleOp module) {
			auto* context = module.getContext();
			if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf")) {
				return SymbolRefAttr::get(context, "printf");
			}

			// Create a function declaration for printf, the signature is:
			//   * `i32 (i8*, ...)`
			auto llvmI32Ty = IntegerType::get(context, 32);
			auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
			auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy, true);

			// Insert the printf function into the body of the parent module.
			PatternRewriter::InsertionGuard insertGuard(rewriter);
			rewriter.setInsertionPointToStart(module.getBody());
			rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
			return SymbolRefAttr::get(context, "printf");
		}

		/// Return a value representing an access into a global string with the given name, creating the string if
		/// necessary.
		static Value getOrCreateGlobalString(Location loc, OpBuilder& builder, StringRef name, StringRef value,
				ModuleOp module) {
			// Create the global at the entry of the module.
			LLVM::GlobalOp global;
			if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
				OpBuilder::InsertionGuard insertGuard(builder);
				builder.setInsertionPointToStart(module.getBody());
				auto type = LLVM::LLVMArrayType::get(IntegerType::get(builder.getContext(), 8), value.size());
				global = builder.create<LLVM::GlobalOp>(loc, type, true, LLVM::Linkage::Internal, name,
						builder.getStringAttr(value));
			}

			// Get the pointer to the first character in the global string.
			Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
			Value cst0 = builder.create<LLVM::ConstantOp>(loc, IntegerType::get(builder.getContext(), 64),
					builder.getIntegerAttr(builder.getIndexType(), 0));
			return builder.create<LLVM::GEPOp>(loc,
					LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)), globalPtr,
					ArrayRef<Value>({cst0, cst0}));
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
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// TvlToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
	struct TvlToLLVMLoweringPass : public PassWrapper<TvlToLLVMLoweringPass, OperationPass<ModuleOp>> {
		void getDependentDialects(DialectRegistry& registry) const final {
			registry.insert<LLVM::LLVMDialect, StandardOpsDialect, scf::SCFDialect>();
		}

		void runOnOperation() final;
	};
} // end anonymous namespace

void TvlToLLVMLoweringPass::runOnOperation() {
	// The first thing to define is the conversion target. This will define the final target for this lowering. For this
	// lowering, we are only targeting the LLVM dialect.
	LLVMConversionTarget target(getContext());
	target.addLegalOp<ModuleOp/*, ModuleTerminatorOp*/>();

	// During this lowering, we will also be lowering the MemRef types, that are currently being operated on, to a
	// representation in LLVM. To perform this conversion we use a TypeConverter as part of the lowering. This converter
	// details how one type maps to another. This is necessary now that we will be doing more complicated lowerings,
	// involving loop region arguments.
	LLVMTypeConverter typeConverter(&getContext());

	// Now that the conversion target has been defined, we need to provide the patterns used for lowering. At this point
	// of the compilation process, we have a combination of `tvl`, `affine`, and `std` operations. Luckily, there are
	// already exists a set of patterns to transform `affine` and `std` dialects. These patterns lowering in multiple
	// stages, relying on transitive lowerings. Transitive lowering, or A->B->C lowering, is when multiple patterns must
	// be applied to fully transform an illegal operation into a set of legal ones.
	RewritePatternSet patterns(&getContext());
	//populateAffineToStdConversionPatterns(patterns, &getContext());
	populateLoopToStdConversionPatterns(patterns);
	populateStdToLLVMConversionPatterns(typeConverter, patterns);

	// The only remaining operation to lower from the `tvl` dialect, is the PrintOp.
	patterns.insert<AddOpLowering, ConstantOpLowering, DivOpLowering, LoadOpLowering, MulOpLowering, PrintOpLowering,
			RemOpLowering, ReturnOpLowering, StoreOpLowering, SubOpLowering>(&getContext());

	// We want to completely lower to LLVM, so we use a `FullConversion`. This ensures that only legal operations will
	// remain after the conversion.
	auto module = getOperation();
	if (failed(applyFullConversion(module, target, std::move(patterns)))) {
		signalPassFailure();
	}
}

/// Create a pass for lowering operations the remaining `TVL` operations, as well as `SCF` and `Std`, to the LLVM
/// dialect for codegen.
std::unique_ptr<Pass> tvl::createLowerToLLVMPass() {
	return std::make_unique<TvlToLLVMLoweringPass>();
}
