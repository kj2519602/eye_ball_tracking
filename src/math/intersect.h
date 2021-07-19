#ifndef _SRC_MATH_INTERSECT_H_
#define _SRC_MATH_INTERSECT_H_

#include <iostream>
#include "Eigen/Dense"

#include "math/geometry.h"

namespace pupil {

class no_intersection_exception : public std::runtime_error {
public:
    no_intersection_exception(const Line3d& line, const Sphere& sphere)
        : std::runtime_error("Line and sphere do not intersect")
    {
    }
};

template<int Dim>
Eigen::Matrix<Scalar, Dim, 1> intersect_lines(
    const std::vector<Eigen::ParametrizedLine<Scalar, Dim>>& lines) {
    using Vector = Eigen::Matrix<Scalar, Dim, 1>;
    using Matrix = Eigen::Matrix<Scalar, Dim, Dim>;

    size_t N = lines.size();   
    std::vector<Matrix> Ivv;
    Matrix A = Matrix::Zero();
    Vector b = Vector::Zero();
    size_t i = 0;
    for (auto& line : lines) {
        Vector vi = line.direction();
        Vector pi = line.origin();

        Matrix Ivivi = Matrix::Identity() - vi * vi.transpose();

        Ivv.push_back(Ivivi);

        A += Ivivi;
        b += (Ivivi * pi);

        i++;
    }
    return A.partialPivLu().solve(b);
}

std::pair<Vector3, Vector3> intersect_line_sphere(const Line3d& line, const Sphere& sphere) {
	if (!(std::abs(line.direction().norm() - 1) < 0.0001)){
        throw no_intersection_exception(line, sphere);
	}

    Vector3 v = line.direction();
    // Put p at origin.
    Vector3 p = line.origin();
    Vector3 c = sphere.center - p;
    Scalar r = sphere.radius;

    Scalar vcvc_cc_rr = sq(v.dot(c)) - c.dot(c) + sq(r);
    if (vcvc_cc_rr < 0) {
        throw no_intersection_exception(line, sphere);
    }

    Scalar s1 = v.dot(c) - std::sqrt(vcvc_cc_rr);
    Scalar s2 = v.dot(c) + std::sqrt(vcvc_cc_rr);

    Vector3 p1 = p + s1*v;
    Vector3 p2 = p + s2*v;

    return std::make_pair(p1, p2);
}

}  // namespace pupil
#endif  // _SRC_MATH_INTERSECT_H_