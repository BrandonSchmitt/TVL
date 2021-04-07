add_mlir_dialect_library(MLIRTvl
	TvlDialect.cpp
	TvlOps.cpp

	ADDITIONAL_HEADER_DIRS
	${PROJECT_SOURCE_DIR}/include/tvl

	DEPENDS
	MLIRTvlOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
)
