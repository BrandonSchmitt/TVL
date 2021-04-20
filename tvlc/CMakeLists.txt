get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRExecutionEngine
        MLIRLLVMToLLVMIRTranslation
        MLIROptLib
        MLIRSupport
        MLIRTvl
        )


bison_target(TvlParser ../lib/tvl/parser.ypp ${PROJECT_BINARY_DIR}/lib/tvl/parser.cpp DEFINES_FILE ${PROJECT_BINARY_DIR}/include/parser.h)
flex_target(TvlScanner ../lib/tvl/scanner.l ${PROJECT_BINARY_DIR}/lib/tvl/scanner.cpp)
ADD_FLEX_BISON_DEPENDENCY(TvlScanner TvlParser)

add_llvm_executable(tvlc
        tvlc.cpp
        ParsingDriver.cpp
        AST.cpp
        TypeInference.cpp
        ../mlir/MLIRGen.cpp
        ../mlir/LowerToLLVM.cpp
        ../mlir/LowerToSCF.cpp
        ../mlir/LowerToStd.cpp
        ${FLEX_TvlScanner_OUTPUTS}
        ${BISON_TvlParser_OUTPUTS}
        )

llvm_update_compile_flags(tvlc)
target_link_libraries(tvlc PRIVATE ${LIBS})

mlir_check_all_link_libraries(tvlc)
