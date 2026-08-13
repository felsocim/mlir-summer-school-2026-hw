// RUN: mlir-opt %s --linalg-specialize-generic-ops | FileCheck %s

//===----------------------------------------------------------------------===//
// EXERCISE 1a - matmul
//
//   C[i, j] += A[i, k] * B[k, j]
//
// Replace every ??? below so that the linalg.generic implements this. Use the
// iteration order (i, j, k).
//
// Reminder of the four things a generic needs:
//   indexing_maps   one map per operand (inputs first, then outputs), each
//                   mapping the iteration space to that operand's indices
//   iterator_types  one entry per iteration space dimension
//   ins/outs        the operands, outputs are destinations that get written
//   the body        scalar computation, ending in linalg.yield
//
// HOW THIS IS CHECKED
// --linalg-specialize-generic-ops recognises well-known computations and folds
// them back into a named op. Get it exactly right and you get a bare
// `linalg.matmul`. Get an indexing map wrong in a way that still verifies (for
// example by transposing B) and you get a `linalg.matmul` that carries an
// explicit `indexing_maps = [...]` attribute, which will not match the CHECK.
//===----------------------------------------------------------------------===//

#mapA = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapB = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapC = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                  %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {
    indexing_maps = [#mapA, #mapB, #mapC],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%C : tensor<?x?xf32>) {
  ^bb0(%in_A: f32, %in_B: f32, %out_C: f32):
    %1 = arith.mulf %in_A, %in_B : f32
    %2 = arith.addf %1, %out_C : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @matmul
// CHECK:         linalg.matmul ins(%arg0, %arg1 :
// CHECK-SAME:      outs(%arg2 :
