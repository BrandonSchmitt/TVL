#include "tvl/TvlDialect.h"
#include "tvl/MLIRGen.h"
#include "tvl/ParsingDriver.h"
#include "tvl/Passes.h"
#include "tvl/TypeInference.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

using namespace llvm;
using namespace tvl;
namespace cl = llvm::cl;

static codegen::RegisterCodeGenFlags codeGenFlags;

static cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input TVL file>"), cl::init("-"),
		cl::value_desc("filename"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"));

namespace {
	enum InputType {
		TVL, MLIR,
	};

	enum ExtensionType {
		NoExtension, AVX512,
	};
}
static cl::opt<enum InputType> inputType("x", cl::init(TVL), cl::desc("Decided the kind of output desired"),
		cl::values(clEnumValN(TVL, "tvl", "load the input file as a TVL source.")),
		cl::values(clEnumValN(MLIR, "mlir", "load the input file as an MLIR file")));

static cl::opt<char> optimizationLevel("O", cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] (default = '-O2')"),
		cl::Prefix, cl::ZeroOrMore, cl::init(' '));

static cl::opt<enum ExtensionType> extensionType("ext", cl::init(NoExtension), cl::desc("Decide which cpu extension to use"),
		cl::values(clEnumValN(NoExtension, "none", "no extension")),
		cl::values(clEnumValN(AVX512, "avx512", "512-bit Advanced Vector Extensions")));

static cl::opt<std::string> targetTriple("mtriple", cl::desc("Override target triple for module"));

namespace {
	enum Action {
		None, DumpAST, DumpMLIR, DumpMLIRSCF, DumpMLIRSTD, DumpMLIRLLVM, DumpLLVMIR, DumpASM, RunJIT
	};
}

static cl::opt<enum Action> emitAction("emit", cl::desc("Select the kind of output desired"),
		cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
		cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
		cl::values(clEnumValN(DumpMLIRSCF, "mlir-scf", "output the MLIR dump after scf lowering")),
		cl::values(clEnumValN(DumpMLIRSTD, "mlir-std", "output the MLIR dump after std lowering")),
		cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm", "output the MLIR dump after llvm lowering")),
		cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
		cl::values(clEnumValN(DumpASM, "asm", "output the target's assembly code")),
		cl::values(clEnumValN(RunJIT, "jit", "JIT the code and run it by invoking the main function")));

LLVM_ATTRIBUTE_NORETURN static void reportError(Twine Msg, StringRef Filename = "") {
	SmallString<256> Prefix;
	if (!Filename.empty()) {
		if (Filename == "-") {
			Filename = "<stdin>";
		}
		("'" + Twine(Filename) + "': ").toStringRef(Prefix);
	}
	WithColor::error(errs()) << Prefix << Msg << "\n";
	exit(1);
}

/// Returns a TVL AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<tvl::ast::Module> parseInputFile(llvm::StringRef filename) {
	llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
	if (std::error_code ec = fileOrErr.getError()) {
		llvm::errs() << "Could not open input file: " << ec.message() << "\n";
		return nullptr;
	}
	auto buffer = fileOrErr.get()->getBuffer();

	tvl::ParsingDriver parsingDriver;
	if (parsingDriver.parse(filename, buffer) != 0) {
		return nullptr;
	}

	auto module = parsingDriver.getModule();
	if (!inferTypes(*module)) {
		return nullptr;
	}
	return module;
}

int loadMLIR(mlir::MLIRContext& context, mlir::OwningModuleRef& module) {
	// Handle '.tvl' input to the compiler.
	if (inputType != InputType::MLIR && !llvm::StringRef(inputFilename).endswith(".mlir")) {
		auto moduleAST = parseInputFile(inputFilename);
		if (!moduleAST) {
			return 6;
		}
		module = mlirGen(context, *moduleAST);
		return !module ? 1 : 0;
	}

	// Otherwise, the input is '.mlir'.
	llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
	if (std::error_code EC = fileOrErr.getError()) {
		llvm::errs() << "Could not open input file: " << EC.message() << "\n";
		return -1;
	}

	// Parse the input mlir.
	llvm::SourceMgr sourceMgr;
	sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
	module = mlir::parseSourceFile(sourceMgr, &context);
	if (!module) {
		llvm::errs() << "Error can't load file " << inputFilename << "\n";
		return 3;
	}
	return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext& context, mlir::OwningModuleRef& module) {
	if (int error = loadMLIR(context, module)) {
		return error;
	}

	mlir::PassManager pm(&context);
	// Apply any generic pass manager command line options and run the pipeline.
	applyPassManagerCLOptions(pm);

	// Check to see what granularity of MLIR we are compiling to.
	bool isLoweringToSCF = emitAction >= Action::DumpMLIRSCF;
	bool isLoweringToStd = emitAction >= Action::DumpMLIRSTD;
	bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

	if (isLoweringToSCF) {
		//mlir::OpPassManager& optPM = pm.nest<mlir::tvl::ForOp>();
		pm.addPass(mlir::tvl::createLowerToSCFPass());
	}

	if (isLoweringToStd) {
		switch (extensionType) {
			case NoExtension:
				pm.addPass(mlir::tvl::createNoExtensionLoweringPass());
				break;
			case AVX512:
				pm.addPass(mlir::tvl::createAVX512LoweringPass());
				break;
		}

		pm.addPass(mlir::tvl::createLowerToStdPass());
	}

	if (isLoweringToLLVM) {
		// Finish lowering the toy IR to the LLVM dialect.
		pm.addPass(mlir::tvl::createLowerToLLVMPass());
	}

	if (mlir::failed(pm.run(*module))) {
		return 4;
	}
	return 0;
}

std::unique_ptr<ToolOutputFile> getOutputStream(const char* targetName, Triple::OSType osType) {
	if (outputFilename.empty()) {
		if (inputFilename == "-") {
			outputFilename = "-";
		} else {
			// If InputFilename ends in .bc or .ll, remove it.
			StringRef filename = inputFilename;

			if (filename.endswith(".bc") || filename.endswith(".ll")) {
				outputFilename = std::string(filename.drop_back(3));
			} else if (filename.endswith(".mir")) {
				outputFilename = std::string(filename.drop_back(4));
			} else {
				outputFilename = std::string(filename);
			}

			switch (codegen::getFileType()) {
				case CGFT_AssemblyFile:
					if (targetName[0] == 'c') {
						if (targetName[1] == 0) {
							outputFilename += ".cbe.c";
						} else if (targetName[1] == 'p' && targetName[2] == 'p') {
							outputFilename += ".cpp";
						} else {
							outputFilename += ".s";
						}
					} else {
						outputFilename += ".s";
					}
					break;
				case CGFT_ObjectFile:
					if (osType == Triple::Win32) {
						outputFilename += ".obj";
					} else {
						outputFilename += ".o";
					}
					break;
				case CGFT_Null:
					outputFilename = "-";
					break;
			}
		}
	}

	// Decide if we need "binary" output.
	bool binary = false;
	switch (codegen::getFileType()) {
		case CGFT_AssemblyFile:
			break;
		case CGFT_ObjectFile:
		case CGFT_Null:
			binary = true;
			break;
	}

	// Open the file.
	std::error_code errorCode;
	sys::fs::OpenFlags openFlags = sys::fs::OF_None;
	if (!binary) {
		openFlags |= sys::fs::OF_TextWithCRLF;
	}
	auto outputFile = std::make_unique<ToolOutputFile>(outputFilename, errorCode, openFlags);
	if (errorCode) {
		reportError(errorCode.message());
		return nullptr;
	}

	return outputFile;
}

static int compileModule(std::unique_ptr<Module> module) {
	// Load the module to be compiled...
	SMDiagnostic Err;
	std::string cpuStr = codegen::getCPUStr();
	std::string featuresStr = codegen::getFeaturesStr();

	auto mAttrs = codegen::getMAttrs();

	CodeGenOpt::Level optLevel = CodeGenOpt::Default;
	switch (optimizationLevel) {
		case ' ':
			break;
		case '0':
			optLevel = CodeGenOpt::None;
			break;
		case '1':
			optLevel = CodeGenOpt::Less;
			break;
		case '2':
			optLevel = CodeGenOpt::Default;
			break;
		case '3':
			optLevel = CodeGenOpt::Aggressive;
			break;
		default:
			WithColor::error(errs()) << "invalid optimization level.\n";
			return 1;
	}

	// Initialize LLVM targets.
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();
	mlir::ExecutionEngine::setupTargetTriple(module.get());

	Triple triple{Triple::normalize(targetTriple)};
	if (targetTriple.empty()) {
		triple.setTriple(module->getTargetTriple());
	}

	Optional<Reloc::Model> relocModel = codegen::getExplicitRelocModel();

	// On AIX, setting the relocation model to anything other than PIC is considered a user error.
	if (triple.isOSAIX() && relocModel.hasValue() && *relocModel != Reloc::PIC_) {
		WithColor::error(errs()) << "invalid relocation model, AIX only supports PIC.\n";
		return 1;
	}

	std::string error;
	const Target* target = TargetRegistry::lookupTarget(codegen::getMArch(), triple, error);
	if (!target) {
		WithColor::error(errs()) << error;
		return 1;
	}

	TargetOptions targetOptions = codegen::InitTargetOptionsFromCodeGenFlags(triple);
	targetOptions.MCOptions.AsmVerbose = true;
	targetOptions.MCOptions.PreserveAsmComments = true;

	auto targetMachine = std::unique_ptr<TargetMachine>(
			target->createTargetMachine(triple.getTriple(), cpuStr, featuresStr, targetOptions, relocModel,
					codegen::getExplicitCodeModel(), optLevel));
	assert(targetMachine && "Could not allocate target machine!");

	// If user just wants to list available options, skip module loading
	if (codegen::getMCPU() == "help" || (!mAttrs.empty() && mAttrs.front() == "help")) {
		return 0;
	}

	if (codegen::getFloatABIForCalls() != FloatABI::Default) {
		targetOptions.FloatABIType = codegen::getFloatABIForCalls();
	}

	// Figure out where we are going to send the output.
	std::unique_ptr<ToolOutputFile> toolOutputFile = getOutputStream(target->getName(), triple.getOS());
	if (!toolOutputFile) {
		return 1;
	}

	// Build up all of the passes that we want to do to the module.
	legacy::PassManager passManager;

	// Add an appropriate TargetLibraryInfo pass for the module's triple.
	TargetLibraryInfoImpl TLII(Triple(module->getTargetTriple()));

	passManager.add(new TargetLibraryInfoWrapperPass(TLII));

	// Override function attributes based on cpuStr, featuresStr, and command line flags.
	codegen::setFunctionAttributes(cpuStr, featuresStr, *module);

	if (mc::getExplicitRelaxAll() && codegen::getFileType() != CGFT_ObjectFile) {
		WithColor::warning(errs()) << ": warning: ignoring -mc-relax-all because filetype != obj";
	}

	{
		raw_pwrite_stream* outputStream = &toolOutputFile->os();

		// Manually do the buffering rather than using buffer_ostream, so we can memcmp the contents in CompileTwice mode
		SmallVector<char, 0> Buffer;
		std::unique_ptr<raw_svector_ostream> bufferOutputStream;
		if ((codegen::getFileType() != CGFT_AssemblyFile && !toolOutputFile->os().supportsSeeking())) {
			bufferOutputStream = std::make_unique<raw_svector_ostream>(Buffer);
			outputStream = bufferOutputStream.get();
		}

		auto& llvmTargetMachine = static_cast<LLVMTargetMachine&>(*targetMachine);
		auto* MMIWP = new MachineModuleInfoWrapperPass(&llvmTargetMachine);

		if (targetMachine->addPassesToEmitFile(passManager, *outputStream, nullptr, codegen::getFileType(), false,
				MMIWP)) {
			reportError("target does not support generation of this file type");
		}

		llvmTargetMachine.getObjFileLowering()->Initialize(MMIWP->getMMI().getContext(), *targetMachine);

		// Before executing passes, print the final values of the LLVM options.
		cl::PrintOptionValues();

		passManager.run(*module);

		if (bufferOutputStream) {
			toolOutputFile->os() << Buffer;
		}
	}

	// Declare success.
	toolOutputFile->keep();

	return 0;
}

int dumpAST() {
	if (inputType == InputType::MLIR) {
		llvm::errs() << "Can't dump a TVL AST when the input is MLIR\n";
		return 5;
	}

	auto moduleAST = parseInputFile(inputFilename);
	if (!moduleAST) {
		return 1;
	}

	dump(*moduleAST);
	return 0;
}

int dumpLLVMIR(mlir::ModuleOp module) {
	// Register the translation to LLVM IR with the MLIR context.
	mlir::registerLLVMDialectTranslation(*module->getContext());

	// Convert the module to LLVM IR in a new LLVM IR context.
	llvm::LLVMContext llvmContext;
	auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
	if (!llvmModule) {
		llvm::errs() << "Failed to emit LLVM IR\n";
		return -1;
	}

	// Initialize LLVM targets.
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();
	mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

	/// Optionally run an optimization pipeline over the llvm module.
	auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
	if (auto err = optPipeline(llvmModule.get())) {
		llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
		return -1;
	}
	llvm::errs() << *llvmModule << "\n";
	return 0;
}

int dumpAsm(mlir::ModuleOp module) {
	// Register the translation to LLVM IR with the MLIR context.
	mlir::registerLLVMDialectTranslation(*module->getContext());

	// Convert the module to LLVM IR in a new LLVM IR context.
	LLVMContext llvmContext;
	auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
	if (!llvmModule) {
		llvm::errs() << "Failed to emit LLVM IR\n";
		return -1;
	}

	return compileModule(std::move(llvmModule));
}

int runJit(mlir::ModuleOp module) {
	// Initialize LLVM targets.
	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	// Register the translation from MLIR to LLVM IR, which must happen before we can JIT-compile.
	mlir::registerLLVMDialectTranslation(*module->getContext());

	// An optimization pipeline to use within the execution engine.
	auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);

	// Create an MLIR execution engine. The execution engine eagerly JIT-compiles the module.
	auto maybeEngine = mlir::ExecutionEngine::create(module, nullptr, optPipeline);
	assert(maybeEngine && "failed to construct an execution engine");
	auto& engine = maybeEngine.get();

	// Invoke the JIT-compiled function.
	auto invocationResult = engine->invokePacked("main");
	if (invocationResult) {
		llvm::errs() << "JIT invocation failed\n";
		return -1;
	}

	return 0;
}

int main(int argc, char** argv) {
	// Register any command line options.
	mlir::registerAsmPrinterCLOptions();
	mlir::registerMLIRContextCLOptions();
	mlir::registerPassManagerCLOptions();
	cl::ParseCommandLineOptions(argc, argv, "TVL compiler\n");

	if (emitAction == Action::DumpAST) {
		return dumpAST();
	}

	// If we aren't dumping the AST, then we are compiling with/to MLIR.

	mlir::MLIRContext context;
	// Load our Dialect in this MLIR Context.
	context.getOrLoadDialect<mlir::tvl::TvlDialect>();
	context.getOrLoadDialect<mlir::memref::MemRefDialect>();

	mlir::OwningModuleRef module;
	if (int error = loadAndProcessMLIR(context, module)) {
		return error;
	}

	// If we aren't exporting to non-mlir, then we are done.
	bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
	if (isOutputingMLIR) {
		module->dump();
		return 0;
	}

	// Check to see if we are compiling to LLVM IR.
	if (emitAction == Action::DumpLLVMIR) {
		return dumpLLVMIR(*module);
	}

	if (emitAction == Action::DumpASM) {
		return dumpAsm(*module);
	}

	// Otherwise, we must be running the jit.
	if (emitAction == Action::RunJIT) {
		return runJit(*module);
	}

	llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
	return -1;
}
