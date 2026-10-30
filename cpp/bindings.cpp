#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "enumerate_plan.h"

namespace py = pybind11;

// pybind11
PYBIND11_MODULE(enumerate_plan, m) {
  // Enums
  py::enum_<ScanMethod>(m, "ScanMethod")
      .value("SeqScan", ScanMethod::SeqScan)
      .value("IndexScan", ScanMethod::IndexScan)
      .value("IndexOnlyScan", ScanMethod::IndexOnlyScan)
      .value("BitmapScan", ScanMethod::BitmapScan);

  py::enum_<JoinMethod>(m, "JoinMethod")
      .value("NestedLoopJoin", JoinMethod::NestedLoopJoin)
      .value("HashJoin", JoinMethod::HashJoin)
      .value("MergeJoin", JoinMethod::MergeJoin);

  // Abstract base class
  py::class_<PlanNode, std::shared_ptr<PlanNode>>(m, "PlanNode")
      .def_property_readonly("relation", &PlanNode::relation)
      .def_property_readonly("left", &PlanNode::left)
      .def_property_readonly("right", &PlanNode::right)
      .def("is_join", &PlanNode::is_join);

  py::class_<ScanNode, PlanNode, std::shared_ptr<ScanNode>>(m, "ScanNode")
      .def(py::init<int, ScanMethod>())
      .def_readwrite("relation", &ScanNode::relation_)
      .def_readwrite("scan_method", &ScanNode::scan_method_);

  py::class_<JoinNode, PlanNode, std::shared_ptr<JoinNode>>(m, "JoinNode")
      .def(py::init<JoinMethod, std::shared_ptr<PlanNode>, std::shared_ptr<PlanNode>>())
      .def_readwrite("join_method", &JoinNode::join_method_)
      .def_readwrite("left", &JoinNode::left_)
      .def_readwrite("right", &JoinNode::right_);

  m.def("enumerate_plan", &enumerate_plan,
        "Enumerate plans from join graph");
}
