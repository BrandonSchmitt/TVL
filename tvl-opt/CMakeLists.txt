get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRTvl
        )
add_llvm_executable(tvl-opt tvl-opt.cpp)

llvm_update_compile_flags(tvl-opt)
target_link_libraries(tvl-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(tvl-opt)
