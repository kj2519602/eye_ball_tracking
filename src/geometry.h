#ifndef _GEOMETRY_H_
#define _GEOMETRY_H_

#include <stdint.h>
#include <cmath>
#include "Eigen/Dense"

#include "opencv2/core/core.hpp"

namespace pupil {

#define M_PI 3.14159265358979323846264338327950288
using Scalar = double;
using Vector2 = Eigen::Matrix<Scalar, 2, 1>;
using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
using Array3 = Eigen::Array<Scalar, 1, 3>;
using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;
using Line2d = Eigen::ParametrizedLine<Scalar, 2>;
using Line3d = Eigen::ParametrizedLine<Scalar, 3>;

inline Scalar sq(Scalar x) {
  return x * x;
}

struct Conic;

struct Ellipse {
// 2D Ellipse.
public:
  Scalar cx, cy;
  Scalar major_radius;
  Scalar minor_radius;
  Scalar angle;

  Ellipse() = default;
  Ellipse(Scalar cx, Scalar cy, Scalar major_radius, Scalar minor_radius, Scalar angle)
      : cx(cx), cy(cy), major_radius(major_radius), minor_radius(minor_radius), angle(angle) {
  }
  explicit Ellipse(const cv::RotatedRect& rect)
      : cx(static_cast<Scalar>(rect.center.x)),
      cy(static_cast<Scalar>(rect.center.y)),
      major_radius(static_cast<Scalar>(rect.size.width / 2)),
      minor_radius(static_cast<Scalar>(rect.size.height / 2)),
      angle(static_cast<Scalar>(rect.angle * M_PI / 180)) {
      }
  explicit Ellipse(const Conic& conic);
  Vector2 Center() const;
  cv::RotatedRect Rect() const;
};

struct Conic {
public:
  Scalar A, B, C, D, E, F;

  Conic() = default;
  Conic(Scalar a, Scalar b, Scalar c, Scalar d, Scalar e, Scalar f)
      : A(a), B(b), C(c), D(d), E(e), F(f) {}
  explicit Conic(const Ellipse& ellipse);

  inline Scalar operator()(Scalar x, Scalar y) const {
      return A*x*x + B*x*y + C*y*y + D*x + E*y + F;
  }
};

struct Circle {
// 3D Circle.
public:
  Vector3 position;
  Vector3 norm;
  Scalar radius;

  Circle() = default;
  Circle(const Vector3& position, const Vector3 norm, Scalar radius)
      : position(std::move(position)), norm(std::move(norm)), radius(std::move(radius)) {
  }
};

using DualCircle = std::pair<Circle, Circle>;

struct Conicoid {
public:
  Scalar A, B, C, F, G, H, U, V, W, D;

  Conicoid() = default;
  Conicoid(Scalar A, Scalar B, Scalar C, Scalar F, Scalar G, Scalar H, Scalar D)
    : A(A), B(B), C(C), F(F), G(G), H(H), U(U), V(V), W(W), D(D) {
  }

  Conicoid(const Conic& conic, const Vector3& vertex);

  inline Scalar operator()(Scalar x, Scalar y, Scalar z) const {
      return A * x * x + B * y * y + C * z * z +
      2 * F * y * z + 2 * G * x * z + 2 * H * x * y + 2 * U * x + 2 * V * y + 2 * W * z + D;
  }

};

struct Sphere {
public:
  Vector3 center;
  Scalar radius;

  Sphere() = default;
  Sphere(Vector3 center, Scalar(radius))
    : center(std::move(center)), radius(radius) {
  }
};

}  // namespace pupil

#endif  // _GEOMETRY_H_