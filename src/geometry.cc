#include "geometry.h"

namespace pupil {

Ellipse::Ellipse(const Conic& conic) {
    angle = 0.5*std::atan2(conic.B, conic.A - conic.C);
    auto cost = std::cos(angle);
    auto sint = std::sin(angle);
    auto sin_squared = sint * sint;
    auto cos_squared = cost * cost;

    auto Ao = conic.F;
    auto Au = conic.D * cost + conic.E * sint;
    auto Av = -conic.D * sint + conic.E * cost;
    auto Auu = conic.A * cos_squared + conic.C * sin_squared + conic.B * sint * cost;
    auto Avv = conic.A * sin_squared + conic.C * cos_squared - conic.B * sint * cost;

    // ROTATED = [Ao Au Av Auu Avv]

    auto tuCentre = -Au / (2.0*Auu);
    auto tvCentre = -Av / (2.0*Avv);
    auto wCentre = Ao - Auu*tuCentre*tuCentre - Avv*tvCentre*tvCentre;

    cx = tuCentre * cost - tvCentre * sint + 160;
    cy = tuCentre * sint + tvCentre * cost + 120;

    major_radius = std::sqrt(std::abs(-wCentre / Auu));
    minor_radius = std::sqrt(std::abs(-wCentre / Avv));
}

Vector2 Ellipse::Center() const {
    return Vector2(cx, cy);
}

cv::RotatedRect Ellipse::Rect() const {
    return cv::RotatedRect(cv::Point2f(cx, cy),
                           cv::Size2f(2*major_radius, 2*minor_radius),
                           angle * 180 / M_PI);
}

Conic::Conic(const Ellipse& ellipse) {
    auto cx = ellipse.cx;
    auto cy = ellipse.cy;
    auto angle = ellipse.angle;
    auto ax = std::cos(angle);
    auto ay = std::sin(angle);
    auto major_radius = ellipse.major_radius;
    auto minor_radius = ellipse.minor_radius;
    auto a2 = major_radius * major_radius;
    auto b2 = minor_radius * minor_radius;

    A = ax*ax / a2 + ay*ay / b2;
    B = 2 * ax*ay / a2 - 2 * ax*ay / b2;
    C = ay*ay / a2 + ax*ax / b2;
    D = (-2 * ax*ay*cy - 2 * ax*ax*cx) / a2 + (2 * ax*ay*cy - 2 * ay*ay*cx) / b2;
    E = (-2 * ax*ay*cx - 2 * ay*ay*cy) / a2 + (2 * ax*ay*cx - 2 * ax*ax*cy) / b2;
    F = (2 * ax*ay*cx*cy + ax*ax*cx * cx + ay*ay*cx*cx) / a2
        + (-2 * ax*ay*cx*cy + ay*ay*cx*cx + ax*ax*cy*cy) / b2 - 1;
}

Conicoid::Conicoid(const Conic& conic, const Vector3& vertex) {
    auto alpha = vertex[0];
    auto beta = vertex[1];
    auto gamma = vertex[2];

    A = sq(gamma) * conic.A;
    B = sq(gamma) * conic.C;
    C = conic.A*sq(alpha) + conic.B*alpha*beta + conic.C*sq(beta) + conic.D*alpha + conic.E*beta + conic.F;
    F = -gamma * (conic.C * beta + conic.B / 2 * alpha + conic.E / 2);
    G = -gamma * (conic.B / 2 * beta + conic.A * alpha + conic.D / 2);
    H = sq(gamma) * conic.B / 2;
    U = sq(gamma) * conic.D / 2;
    V = sq(gamma) * conic.E / 2;
    W = -gamma * (conic.E / 2 * beta + conic.D / 2 * alpha + conic.F);
    D = sq(gamma) * conic.F;
}

}  // namespace pupil