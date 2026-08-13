// RUN: mlir-opt %s --linalg-specialize-generic-ops | FileCheck %s

//===----------------------------------------------------------------------===//
// EXERCISE 1b - matmul followed by an elementwise add
//
//   D[i, j] = (sum over k of A[i, k] * B[k, j]) + C[i, j]
//
// This is the running example of the lecture and the payload you will schedule
// in exercise 4.
//
// Before you start: how many linalg operations does this need? Try to write it
// as one, then read the note at the bottom of this file.
//
// %acc arrives already zeroed and %out is the destination of the add, so you do
// not have to deal with linalg.fill here.
//===----------------------------------------------------------------------===//

#mapA = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapB = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapC = affine_map<(d0, d1, d2) -> (d0, d1)>
#mapD = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_add(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
                      %C: tensor<?x?xf32>, %acc: tensor<?x?xf32>,
                      %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // Step 1: %mm = A * B, accumulated into %acc.
  %mm = linalg.generic {
    indexing_maps = [#mapA, #mapB, #mapC],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%acc : tensor<?x?xf32>) {
  ^bb0(%in_A: f32, %in_B: f32, %out_acc: f32):
    %1 = arith.mulf %in_A, %in_B : f32
    %2 = arith.addf %1, %out_acc : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>

  // Step 2: %d = %mm + C, written into %out.
  %d = linalg.generic {
    indexing_maps = [#mapD, #mapD, #mapD],
    iterator_types = ["parallel", "parallel"]
  } ins(%mm, %C : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%out : tensor<?x?xf32>) {
  ^bb0(%in_mm: f32, %in_C: f32, %out_D: f32):
    %1 = arith.addf %in_mm, %in_C : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>

  return %d : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @matmul_add
// CHECK:         linalg.matmul ins(
// CHECK:         linalg.add ins(

//===----------------------------------------------------------------------===//
// WHY NOT ONE OPERATION?
//
// A generic has a single iteration space and its body runs once per point in
// it. A matmul needs the space (i, j, k), so the body runs K times for every
// output element. Adding C[i, j] inside that body would therefore add C once
// per k, that is K times instead of once.
//
// The reduction has to finish before the add can happen, which is exactly what
// makes these two operations rather than one. Keeping them separate is not a
// loss: in exercise 4 you will tile the add and fuse the matmul into it, which
// recovers the locality of a single fused loop nest while leaving the algorithm
// written the obvious way.
//===----------------------------------------------------------------------===//
