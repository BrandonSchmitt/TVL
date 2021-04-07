#ifndef TVL_OPS_H
#define TVL_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "tvl/TvlOps.h.inc"

#endif // TVL_OPS_H
