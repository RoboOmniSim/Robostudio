#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "barycentric.cuh"

namespace py = pybind11;

void compute_barycentric(py::array_t<float> point, py::array_t<float> tetrahedron, py::array_t<float> barycentric_coords) {
    auto p = point.unchecked<1>();
    auto t = tetrahedron.unchecked<2>();
    auto b = barycentric_coords.mutable_unchecked<1>();

    Point3D point3D = { p(0), p(1), p(2) };
    Point3D tetrahedron_points[4];
    for (int i = 0; i < 4; ++i) {
        tetrahedron_points[i] = { t(i, 0), t(i, 1), t(i, 2) };
    }

    float barycentric[4];
    computeBarycentric(point3D, tetrahedron_points, barycentric);

    for (int i = 0; i < 4; ++i) {
        b(i) = barycentric[i];
    }
}

PYBIND11_MODULE(barycentric, m) {
    m.def("compute_barycentric", &compute_barycentric, "Compute barycentric coordinates",
          py::arg("point"), py::arg("tetrahedron"), py::arg("barycentric_coords"));
}
