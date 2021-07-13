#include "tvl/TvlDialect.h"
#include "tvl/TvlOps.h"
#include "tvl/Passes.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <ctime>

using namespace mlir;

namespace {
	class InstantNowOpLowering : public ConversionPattern {
	public:
		explicit InstantNowOpLowering(MLIRContext* context)
				: ConversionPattern(tvl::InstantNowOp::getOperationName(), 1, context) {}

		LogicalResult
		matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			auto loc = op->getLoc();

			auto parentModule = op->getParentOfType<ModuleOp>();

			// Get a symbol reference to the printf function, inserting it if necessary.
			auto clockGetTimeRef = getOrInsertClockGetTime(rewriter, parentModule);

			//auto randOp = cast<tvl::RandOp>(op);
			//auto high = rewriter.create<CallOp>(loc, clockGetTimeRef, rewriter.getI32Type())->getOpResult(0);
			//auto highCast = rewriter.create<ZeroExtendIOp>(loc, high, rewriter.getI64Type());
			//auto low = rewriter.create<CallOp>(loc, randRef, rewriter.getI32Type())->getOpResult(0);
			//auto lowCast = rewriter.create<ZeroExtendIOp>(loc, low, rewriter.getI64Type());
			//auto shiftValue = rewriter.create<ConstantOp>(loc, rewriter.getI64IntegerAttr(32));
			//auto shiftedHigh = rewriter.create<ShiftLeftOp>(loc, highCast, shiftValue)->getOpResult(0);
			//rewriter.replaceOpWithNewOp<OrOp>(op, shiftedHigh, lowCast);
			Value clockId = rewriter.create<ConstantOp>(loc, rewriter.getI32IntegerAttr(CLOCK_MONOTONIC));

			auto* context = parentModule.getContext();
			/*auto llvmI64Ty = IntegerType::get(context, 64);
			ArrayRef<mlir::Type> elementTypes{llvmI64Ty, llvmI64Ty};
			auto structType = LLVM::LLVMStructType::getLiteral(context, elementTypes, true);

			//auto memRefType = mlir::MemRefType::get({1}, structType);*/
			ArrayRef<mlir::Type> elementTypes{rewriter.getI64Type(), rewriter.getI64Type()};
			auto memRefType = mlir::MemRefType::get({1}, rewriter.getType<tvl::StructType>(elementTypes));
			Value memref = rewriter.create<memref::AllocaOp>(loc, memRefType);
			//Value timespecStruct = rewriter.create<ConstantOp>(loc, rewriter.ge)
			rewriter.replaceOpWithNewOp<CallOp>(op, clockGetTimeRef, rewriter.getI32Type(), ArrayRef<Value>({clockId, memref}));

			return success();
		}

		/// Return a symbol reference to the printf function, inserting it into the module if necessary.
		static FlatSymbolRefAttr getOrInsertClockGetTime(PatternRewriter& rewriter, ModuleOp module) {
			auto* context = module.getContext();
			if (module.lookupSymbol<LLVM::LLVMFuncOp>("clock_gettime")) {
				return SymbolRefAttr::get(context, "clock_gettime");
			}

			auto llvmI32Ty = IntegerType::get(context, 32);
			auto llvmI64Ty = IntegerType::get(context, 64);
			ArrayRef<mlir::Type> elementTypes{llvmI64Ty, llvmI64Ty};
			auto structType = LLVM::LLVMStructType::getLiteral(context, elementTypes, true);
			auto structPtrType = LLVM::LLVMPointerType::get(structType);
			auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, {llvmI32Ty, structPtrType});

			// Insert the rand function into the body of the parent module.
			PatternRewriter::InsertionGuard insertGuard(rewriter);
			rewriter.setInsertionPointToStart(module.getBody());
			rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "clock_gettime", llvmFnType);
			return SymbolRefAttr::get(context, "clock_gettime");
		}
	};

	class InstantElapsedOpLowering : public ConversionPattern {
	public:
		explicit InstantElapsedOpLowering(MLIRContext* context)
				: ConversionPattern(tvl::InstantElapsedOp::getOperationName(), 1, context) {}

		LogicalResult
		matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			auto loc = op->getLoc();

			auto parentModule = op->getParentOfType<ModuleOp>();

			// Get a symbol reference to the printf function, inserting it if necessary.
			auto clockGetTimeRef = InstantNowOpLowering::getOrInsertClockGetTime(rewriter, parentModule);

			//auto randOp = cast<tvl::RandOp>(op);
			//auto high = rewriter.create<CallOp>(loc, clockGetTimeRef, rewriter.getI32Type())->getOpResult(0);
			//auto highCast = rewriter.create<ZeroExtendIOp>(loc, high, rewriter.getI64Type());
			//auto low = rewriter.create<CallOp>(loc, randRef, rewriter.getI32Type())->getOpResult(0);
			//auto lowCast = rewriter.create<ZeroExtendIOp>(loc, low, rewriter.getI64Type());
			//auto shiftValue = rewriter.create<ConstantOp>(loc, rewriter.getI64IntegerAttr(32));
			//auto shiftedHigh = rewriter.create<ShiftLeftOp>(loc, highCast, shiftValue)->getOpResult(0);
			//rewriter.replaceOpWithNewOp<OrOp>(op, shiftedHigh, lowCast);
			rewriter.replaceOpWithNewOp<CallOp>(op, clockGetTimeRef, rewriter.getI32Type());

			return success();
		}
	};

	class PrintOpLowering : public ConversionPattern {
	public:
		explicit PrintOpLowering(MLIRContext* context)
				: ConversionPattern(tvl::PrintOp::getOperationName(), 1, context) {}

		LogicalResult
		matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			auto printOp = cast<tvl::PrintOp>(op);
			auto loc = op->getLoc();

			auto parentModule = op->getParentOfType<ModuleOp>();

			// Get a symbol reference to the printf function, inserting it if necessary.
			auto printfRef = getOrInsertPrintf(rewriter, parentModule);

			rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32), operands);

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

	class RandOpLowering : public ConversionPattern {
	public:
		explicit RandOpLowering(MLIRContext* context)
				: ConversionPattern(tvl::RandOp::getOperationName(), 1, context) {}

		LogicalResult
		matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			auto loc = op->getLoc();

			ModuleOp parentModule = op->getParentOfType<ModuleOp>();

			// Get a symbol reference to the printf function, inserting it if necessary.
			auto randRef = getOrInsertRand(rewriter, parentModule);

			//auto randOp = cast<tvl::RandOp>(op);
			auto high = rewriter.create<CallOp>(loc, randRef, rewriter.getI32Type())->getOpResult(0);
			auto highCast = rewriter.create<ZeroExtendIOp>(loc, high, rewriter.getI64Type());
			auto low = rewriter.create<CallOp>(loc, randRef, rewriter.getI32Type())->getOpResult(0);
			auto lowCast = rewriter.create<ZeroExtendIOp>(loc, low, rewriter.getI64Type());
			auto shiftValue = rewriter.create<ConstantOp>(loc, rewriter.getI64IntegerAttr(32));
			auto shiftedHigh = rewriter.create<ShiftLeftOp>(loc, highCast, shiftValue)->getOpResult(0);
			rewriter.replaceOpWithNewOp<OrOp>(op, shiftedHigh, lowCast);

			return success();
		}

	private:
		/// Return a symbol reference to the printf function, inserting it into the module if necessary.
		static FlatSymbolRefAttr getOrInsertRand(PatternRewriter& rewriter, ModuleOp module) {
			auto* context = module.getContext();
			if (module.lookupSymbol<LLVM::LLVMFuncOp>("rand")) {
				return SymbolRefAttr::get(context, "rand");
			}

			auto llvmI32Ty = IntegerType::get(context, 32);
			auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, {});

			// Insert the rand function into the body of the parent module.
			PatternRewriter::InsertionGuard insertGuard(rewriter);
			rewriter.setInsertionPointToStart(module.getBody());
			rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "rand", llvmFnType);
			return SymbolRefAttr::get(context, "rand");
		}
	};

	class SRandOpLowering : public ConversionPattern {
	public:
		explicit SRandOpLowering(MLIRContext* context)
				: ConversionPattern(tvl::SRandOp::getOperationName(), 1, context) {}

		LogicalResult
		matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const final {
			auto loc = op->getLoc();

			ModuleOp parentModule = op->getParentOfType<ModuleOp>();

			auto srandRef = getOrInsertSRand(rewriter, parentModule);

			auto srandOp = cast<tvl::SRandOp>(op);
			rewriter.create<CallOp>(loc, srandRef, LLVM::LLVMVoidType::get(parentModule.getContext()), srandOp.seed());
			rewriter.eraseOp(op);

			return success();
		}

	private:
		/// Return a symbol reference to the printf function, inserting it into the module if necessary.
		static FlatSymbolRefAttr getOrInsertSRand(PatternRewriter& rewriter, ModuleOp module) {
			auto* context = module.getContext();
			if (module.lookupSymbol<LLVM::LLVMFuncOp>("srand")) {
				return SymbolRefAttr::get(context, "srand");
			}

			auto llvmVoidTy = LLVM::LLVMVoidType::get(context);
			auto llvmI32Ty = IntegerType::get(context, 32);
			auto llvmFnType = LLVM::LLVMFunctionType::get(llvmVoidTy, llvmI32Ty);

			// Insert the rand function into the body of the parent module.
			PatternRewriter::InsertionGuard insertGuard(rewriter);
			rewriter.setInsertionPointToStart(module.getBody());
			rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "srand", llvmFnType);
			return SymbolRefAttr::get(context, "srand");
		}
	};

	struct TvlToLLVMLoweringPass : public PassWrapper<TvlToLLVMLoweringPass, OperationPass<ModuleOp>> {
		void getDependentDialects(DialectRegistry& registry) const final {
			registry.insert<LLVM::LLVMDialect>();
		}

		void runOnOperation() final;
	};
} // end anonymous namespace

void TvlToLLVMLoweringPass::runOnOperation() {
	LLVMConversionTarget target(getContext());
	target.addLegalOp<ModuleOp/*, ModuleTerminatorOp*/>();

	// During this lowering, we will also be lowering the MemRef types, that are currently being operated on, to a
	// representation in LLVM. To perform this conversion we use a TypeConverter as part of the lowering. This converter
	// details how one type maps to another. This is necessary now that we will be doing more complicated lowerings,
	// involving loop region arguments.
	LLVMTypeConverter typeConverter(&getContext());

	RewritePatternSet patterns(&getContext());
	populateVectorToLLVMConversionPatterns(typeConverter, patterns);
	populateStdToLLVMConversionPatterns(typeConverter, patterns);

	// The only remaining operation to lower from the `tvl` dialect, is the PrintOp.
	patterns.insert<PrintOpLowering, RandOpLowering, SRandOpLowering, InstantNowOpLowering, InstantElapsedOpLowering>(
			&getContext());

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
