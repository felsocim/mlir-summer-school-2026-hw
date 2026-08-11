//===- Exercise_1_ListLowerFromElements.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the `list-lower-from-elements` pass.
//
//===----------------------------------------------------------------------===//

#include "list/Transforms/ListPasses.h"

#include "list/IR/List.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace list {

#define GEN_PASS_DEF_LISTLOWERFROMELEMENTS
#include "list/Transforms/ListPasses.h.inc"

namespace {

struct ListLowerFromElementsPass
    : public impl::ListLowerFromElementsBase<ListLowerFromElementsPass> {

  void runOnOperation() override {
    SmallVector<FromElementsOp> ops;
    getOperation()->walk([&](FromElementsOp op) { ops.push_back(op); });

    for (auto operation : ops) {
      OpBuilder builder(operation);
      Location location = operation.getLoc();
      Type type = operation.getResult().getType();
      Value empty = list::EmptyOp::create(builder, location, type);
      for (auto item : operation.getElements()) {
        empty = list::PushBackOp::create(builder, location, type, empty, item);
      }
      operation.getResult().replaceAllUsesWith(empty);
      operation.erase();
    }
  }
};

} // namespace

} // namespace list
} // namespace mlir
