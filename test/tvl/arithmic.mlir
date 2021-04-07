// RUN: tvl-translate --import-tvl %s | FileCheck %s

fn main() {
    // CHECK: %[[CONST6:.*]] = tvl.constant 6 : i64
    // CHECK: %[[CONST5:.*]] = tvl.constant 5 : i64
    // CHECK: %[[CONST4:.*]] = tvl.constant 4 : i64
    // CHECK: %[[CONST3:.*]] = tvl.constant 3 : i64
    // CHECK: %[[DIFF:.*]] = tvl.sub %[[CONST4]], %[[CONST3]] : i64
    // CHECK: %[[PROD:.*]] = tvl.mul %[[CONST5]], %[[DIFF]] : i64
    // CHECK: %[[CONST2:.*]] = tvl.constant 2 : i64
    // CHECK: %[[QUOT:.*]] = tvl.div %[[PROD]], %[[CONST2]] : i64
    // CHECK: %[[CONST1:.*]] = tvl.constant 1 : i64
    // CHECK: %[[REM:.*]] = tvl.rem %[[QUOT]], %[[CONST1]] : i64
    // CHECK: %[[SUM:.*]] = tvl.add %[[CONST6]], %[[REM]] : i64

    6 + 5 * (4 - 3) / 2 % 1;
}
