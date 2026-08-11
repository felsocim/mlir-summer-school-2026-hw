// RUN: tutorial-opt --split-input-file %s | FileCheck %s

func.func @empty() -> !list.list<i32> {
  %0 = list.empty : !list.list<i32>
  return %0 : !list.list<i32>
}

// CHECK-LABEL: func.func @empty
// CHECK: %[[LIST:.*]] = list.empty : !list.list<i32>
// CHECK: return %[[LIST]] : !list.list<i32>

// -----

func.func @push_back(%item: i32) -> !list.list<i32> {
  %empty = list.empty : !list.list<i32>
  %list = list.push_back %empty, %item : !list.list<i32>
  return %list : !list.list<i32>
}

// CHECK-LABEL: func.func @push_back
// CHECK: %[[EMPTY:.*]] = list.empty : !list.list<i32>
// CHECK: %[[LIST:.*]] = list.push_back %[[EMPTY]], %{{.*}} : !list.list<i32>
// CHECK: return %[[LIST]] : !list.list<i32>

// -----

func.func @length(%list: !list.list<i32>) -> i32 {
  %length = list.length %list : !list.list<i32> -> i32
  return %length : i32
}

// CHECK-LABEL: func.func @length
// CHECK: %[[LENGTH:.*]] = list.length %{{.*}} : !list.list<i32> -> i32
// CHECK: return %[[LENGTH]] : i32

// -----

func.func @is_empty(%list: !list.list<i32>) -> i1 {
  %empty = list.is_empty %list : !list.list<i32> -> i1
  return %empty : i1
}

// CHECK-LABEL: func.func @is_empty
// CHECK: %[[IS_EMPTY:.*]] = list.is_empty %{{.*}} : !list.list<i32> -> i1
// CHECK: return %[[IS_EMPTY]] : i1

// -----

func.func @print(%list: !list.list<i32>) {
  list.print %list : !list.list<i32>
  return
}

// CHECK-LABEL: func.func @print
// CHECK: list.print %{{.*}} : !list.list<i32>

// -----

func.func @range(%lower: i32, %upper: i32) -> !list.list<i32> {
  %list = list.range %lower to %upper : !list.list<i32>
  return %list : !list.list<i32>
}

// CHECK-LABEL: func.func @range
// CHECK: %[[LIST:.*]] = list.range %{{.*}} to %{{.*}} : !list.list<i32>
// CHECK: return %[[LIST]] : !list.list<i32>
