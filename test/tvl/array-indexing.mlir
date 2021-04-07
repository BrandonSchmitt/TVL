// RUN: tvl-translate --import-tvl %s | FileCheck %s

fn main() {
    // CHECK: %[[MEMREF:.*]] = memref.alloca() : memref<1xi64>
    // CHECK: tvl.constant 0 : index
    // CHECK: %[[INDEX:.*]] = tvl.constant 0 : index
    // CHECK-NEXT: %[[VALUE:.*]] = tvl.load %[[MEMREF]][%[[INDEX]]] : memref<1xi64>

    let u64 a = [4711];
    a[0];
}
