//===- tvl-translate.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"

#include "tvl/MLIRGen.h"
#include "tvl/ParsingDriver.h"
#include "tvl/TvlDialect.h"

// TODO: maybe change
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
	OwningModuleRef translateTvl(llvm::StringRef source, MLIRContext* context) {
		::tvl::ParsingDriver parsingDriver;
		if (parsingDriver.parse("-", source) != 0) {
			return nullptr;
		}
		auto moduleAST = parsingDriver.getModule();

		context->getOrLoadDialect<mlir::tvl::TvlDialect>();
		context->getOrLoadDialect<mlir::memref::MemRefDialect>();
		return ::tvl::mlirGen(*context, *moduleAST);
	}

	void registerTvlToMLIRRegistration() {
		//std::function<OwningModuleRef(llvm::StringRef, MLIRContext*)> f = [] { return nullptr; };
		TranslateToMLIRRegistration Unused("import-tvl", translateTvl);
	}
}    // namespace mlir

int main(int argc, char** argv) {
	mlir::registerAllTranslations();
	mlir::registerTvlToMLIRRegistration();

	return failed(mlir::mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
