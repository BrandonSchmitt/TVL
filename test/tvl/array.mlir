// RUN: tvl-translate --import-tvl %s | FileCheck %s

fn main() {
    // CHECK: %[[MEMREF:.*]] = memref.alloca() : memref<1xi64>
    // CHECK-NEXT: %[[INDEX:.*]] = tvl.constant 0 : index
    // CHECK-NEXT: %[[VALUE:.*]] = tvl.constant 4711 : i64
    // CHECK-NEXT: memref.store %[[VALUE]], %[[MEMREF]][%[[INDEX]]]

    [4711];
}
