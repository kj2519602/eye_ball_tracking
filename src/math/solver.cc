#include "math/solver.h"

namespace pupil {

Scalar solver(Scalar a, Scalar b) {
    if (a==0) {
        assert(b==0 && "No solution");
        return 0;
    }
    return -b / a;
 }

// ax^2 + bx + c = 0
Vector2 solver(Scalar a, Scalar b, Scalar c) {
    if (a==0) {
        auto root = solver(b, c);
        return Vector2(root, root);
    }
    auto det = b*b - 4 * a*c;
    assert(det >= 0 && "No solution");
    auto sqrtdet = std::sqrt(det);
    auto q = -0.5 * (b + (b >= 0 ? 1 : -1) * std::sqrt(det));
    return Vector2(q / a, c / q);
}

Vector3 solver(Scalar a, Scalar b, Scalar c, Scalar d) {
    if (a == 0) {
        auto roots = solver(b, c, d);
        return Vector3(roots(0), roots(1), roots(1));
    }

    auto p = b / a;
    auto q = c / a;
    auto r = d / a;

    auto u = q - p*p / 3;
    auto v = r - p*q / 3 + 2 * p*p*p / 27;

    auto j = 4 * u*u*u / 27 + v*v;

    const auto M = std::numeric_limits<Scalar>::max();
    const auto sqrtM = std::sqrt(M);
    const auto cbrtM = std::cbrt(M);

    if (b == 0 && c == 0)
        return Vector3(std::cbrt(-d), std::cbrt(-d), std::cbrt(-d));
    if (std::abs(p) > 27 * cbrtM)
        return Vector3(-p, -p, -p);
    if (std::abs(q) > sqrtM)
        return Vector3(-std::cbrt(v), -std::cbrt(v), -std::cbrt(v));
    if (std::abs(u) > 3 * cbrtM / 4)
        return Vector3(std::cbrt(4)*u / 3, std::cbrt(4)*u / 3, std::cbrt(4)*u / 3);

    if (j > 0) {
        // One real root
        auto w = std::sqrt(j);
        Scalar y;
        if (v > 0)
            y = (u / 3)*std::cbrt(2 / (w + v)) - std::cbrt((w + v) / 2) - p / 3;
        else
            y = std::cbrt((w - v) / 2) - (u / 3)*std::cbrt(2 / (w - v)) - p / 3;
        return Vector3(y, y, y);
    }
    else {
        // Three real roots
        auto s = std::sqrt(-u / 3);
        auto t = -v / (2 * s*s*s);
        auto k = std::acos(t) / 3;

        auto y1 = 2 * s*std::cos(k) - p / 3;
        auto y2 = s*(-std::cos(k) + std::sqrt(3.)*std::sin(k)) - p / 3;
        auto y3 = s*(-std::cos(k) - std::sqrt(3.)*std::sin(k)) - p / 3;

        return Vector3(y1, y2, y3);
    }
}

}  // namespace pupil