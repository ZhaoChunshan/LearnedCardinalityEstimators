#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "enumerate_plan.h"

namespace py = pybind11;

// pybind11
PYBIND11_MODULE(enumerate_plan, m) {
  // Abstract base class
  py::class_<PlanNode, std::shared_ptr<PlanNode>>(m, "PlanNode")
      .def_property_readonly("relation", &PlanNode::relation)
      .def_property_readonly("left", &PlanNode::left)
      .def_property_readonly("right", &PlanNode::right)
      .def("is_join", &PlanNode::is_join);

  py::class_<Scan, PlanNode, std::shared_ptr<Scan>>(m, "Scan")
      .def(py::init<int>())
      .def_readwrite("relation", &Scan::relation_);

  py::class_<Join, PlanNode, std::shared_ptr<Join>>(m, "Join")
      .def(py::init<std::shared_ptr<PlanNode>, std::shared_ptr<PlanNode>>())
      .def_readwrite("left", &Join::left_)
      .def_readwrite("right", &Join::right_);

  m.def("enumerate_plan", &enumerate_plan,
        "Enumerate join orders from a join graph");
}
