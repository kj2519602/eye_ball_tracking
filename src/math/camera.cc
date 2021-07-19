#include "math/camera.h"

#include <iostream>
#include <cmath>
#include "math/solver.h"

namespace pupil {

Vector2 Camera::Project(const Vector3& point) const {
    auto x = point(0) * focal_length_ / point(2) + cx_;
    auto y = point(1) * focal_length_ / point(2) + cy_;
    return Vector2({x, y});
}

Line2d Camera::Project(const Circle& circle) const {
  auto origin = Project(circle.position);
  auto direction = Project(circle.position + circle.norm) - origin;
  return Line2d(origin, direction.normalized());
}

Ellipse Camera::Project(const Sphere& sphere) const {
    Ellipse ellipse;
    ellipse.cx = focal_length_ * sphere.center[0] / sphere.center[2] + cx_;
    ellipse.cy = focal_length_ * sphere.center[1] / sphere.center[2] + cy_;
    auto radius = focal_length_ * sphere.radius / sphere.center[2];
    ellipse.major_radius = radius;
    ellipse.minor_radius = radius;
    ellipse.angle = 0.;
    return ellipse;
}

Conic Camera::ProjectCircle(const Circle& circle) const {
    Vector3 c = circle.position;
    Vector3 n = circle.norm;
    Scalar r = circle.radius;
    Scalar f = focal_length_;
    Scalar cn = c.dot(n);
    Scalar c2r2 = (c.dot(c) - sq(r));

    Vector3 ABC = (sq(cn) - 2.0*cn*c.array()*n.array() + c2r2*n.array().square());
    Scalar F = 2.0*(c2r2*n(1)*n(2) - cn*(n(1)*c(2) + n(2)*c(1)));
    Scalar G = 2.0*(c2r2*n(2)*n(0) - cn*(n(2)*c(0) + n(0)*c(2)));
    Scalar H = 2.0*(c2r2*n(0)*n(1) - cn*(n(0)*c(1) + n(1)*c(0)));
    return Conic(ABC(0), H, ABC(1), G*f, F*f, ABC(2)*sq(f));
}

// std::pair<cv::Point2f, cv::Point2f> Camera::Project(const Circle& circle, Scalar length) const {
//   cv::Point2f pixel_1 = Project(circle.position);
//   cv::Point2f pixel_2 = Project(circle.position + circle.norm*length);
//   return std::make_pair(pixel_1, pixel_2);
// }

bool Camera::Unproject(const Ellipse& ellipse, const Scalar radius, DualCircle* dual_circle) const {
    using Translation3 = Eigen::Translation<Scalar, 3>;

    Vector3 cam_centre(0., 0., -focal_length_);
    Ellipse ellipse_shift(ellipse);
    ellipse_shift.cx -= cx_;
    ellipse_shift.cy -= cy_;

    Conicoid pupil_cone(Conic(ellipse_shift), cam_centre);
    auto a = pupil_cone.A;
    auto b = pupil_cone.B;
    auto c = pupil_cone.C;
    auto f = pupil_cone.F;
    auto g = pupil_cone.G;
    auto h = pupil_cone.H;
    auto u = pupil_cone.U;
    auto v = pupil_cone.V;
    auto w = pupil_cone.W;
    auto d = pupil_cone.D;

    auto lambda = Array3(solver(
        1., -(a + b + c), (b*c + c*a + a*b - f*f - g*g - h*h),
        -(a*b*c + 2*f*g*h - a*f*f - b*g*g - c*h*h)));

    if (lambda(0) < lambda(1)) return false;
    if (lambda(1) <= 0) return false;
    if (lambda(2) >= 0) return false;

    auto n = std::sqrt((lambda(1) - lambda(2)) / (lambda(0) - lambda(2)));
    auto m = 0.0;
    auto l = std::sqrt((lambda(0) - lambda(1)) / (lambda(0) - lambda(2)));

    Matrix3 T1;
    auto li = T1.row(0);
    auto mi = T1.row(1);
    auto ni = T1.row(2);

    Array3 t1 = (b - lambda)*g - f*h;
    Array3 t2 = (a - lambda)*f - g*h;
    Array3 t3 = -(a - lambda)*(t1 / t2) / g - h / g;

    mi = 1 / (1 + (t1 / t2).square() + t3.square()).sqrt();
    li = (t1 / t2) * mi.array();
    ni = t3 * mi.array();

    if ((li.cross(mi)).dot(ni) < 0) {
        li = -li;
        mi = -mi;
        ni = -ni;
    }

    Translation3 T2;
    T2.translation() = -(u*li + v*mi + w*ni).array() / lambda;

    Circle solutions[2];
    Scalar ls[2] = { l, -l };
    for (int i = 0; i < 2; i++) {
        auto l = ls[i];
        Vector3 gaze = T1 * Vector3(l, m, n);
        Matrix3 T3;
        if (l == 0) {
            assert(n == 1);
            T3 << 0, -1, 0,
                1, 0, 0,
                0, 0, 1;
        }
        else {
            auto sgnl = sign(l);
            T3 << 0, -n*sign(l), l,
                sign(l), 0, 0,
                0, abs(l), n;
        }

        auto A = lambda.matrix().dot(T3.col(0).cwiseAbs2());
        auto B = lambda.matrix().dot(T3.col(0).cwiseProduct(T3.col(2)));
        auto C = lambda.matrix().dot(T3.col(1).cwiseProduct(T3.col(2)));
        auto D = lambda.matrix().dot(T3.col(2).cwiseAbs2());

        Vector3 centre_in_Xprime;
        centre_in_Xprime(2) = A*radius / std::sqrt(B*B + C*C - A*D);
        centre_in_Xprime(0) = -B / A * centre_in_Xprime(2);
        centre_in_Xprime(1) = -C / A * centre_in_Xprime(2);

        Translation3 T0;
        T0.translation() << 0, 0, focal_length_;

        Vector3 centre = T0*T1*T2*T3*centre_in_Xprime;
        if (centre(2) < 0) {
            centre_in_Xprime = -centre_in_Xprime;
            centre = T0*T1*T2*T3*centre_in_Xprime;
        }
        if (gaze.dot(centre) > 0) {
            gaze = -gaze;
        }
        gaze.normalize();
        solutions[i] = Circle(centre, gaze, radius);
    }
    dual_circle->first = solutions[0];
    dual_circle->second = solutions[1];
    return true;
}

bool Camera::Unproject(const Vector2& point2d, const Scalar distance, Vector3* point3d) const {
    point3d->x() = (point2d(0) - cx_) * distance / focal_length_;
    point3d->y() = (point2d(1) - cy_) * distance / focal_length_;
    point3d->z() = distance;
    return true;
}

}  // namespace pupil