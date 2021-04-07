#include "tvl/TvlDialect.h"
#include "tvl/TvlOps.h"

using namespace mlir;
using namespace mlir::tvl;

//===----------------------------------------------------------------------===//
// TVL dialect.
//===----------------------------------------------------------------------===//

void TvlDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tvl/TvlOps.cpp.inc"
      >();
}
