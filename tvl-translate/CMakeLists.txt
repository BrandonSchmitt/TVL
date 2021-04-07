set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

bison_target(TvlParser ../lib/tvl/parser.ypp ${PROJECT_BINARY_DIR}/lib/tvl/parser.cpp DEFINES_FILE ${PROJECT_BINARY_DIR}/include/parser.h)
flex_target(TvlScanner ../lib/tvl/scanner.l ${PROJECT_BINARY_DIR}/lib/tvl/scanner.cpp)
ADD_FLEX_BISON_DEPENDENCY(TvlScanner TvlParser)

add_llvm_executable(tvl-translate
        tvl-translate.cpp
        ../tvlc/ParsingDriver.cpp
        ../mlir/MLIRGen.cpp
        ${FLEX_TvlScanner_OUTPUTS}
        ${BISON_TvlParser_OUTPUTS}
  )
llvm_update_compile_flags(tvl-translate)
target_link_libraries(tvl-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRTranslation
  MLIRSupport
  )

mlir_check_link_libraries(tvl-translate)
