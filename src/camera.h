#ifndef _SRC_CAMERA_H_
#define _SRC_CAMERA_H_

#include "geometry.h"

namespace pupil {

class Camera {
  public:
    Camera() = default;
    Camera(Scalar focal_length, Scalar cx, Scalar cy, Scalar size_x, Scalar size_y)
     : focal_length_(focal_length), cx_(cx), cy_(cy), size_x_(size_x), size_y_(size_y) {
     }
    explicit Camera(const std::string& filename);

    inline Scalar FocalLength() const { return focal_length_; }
    inline Scalar CenterX() const { return cx_; }
    inline Scalar CenterY() const { return cy_; }
    inline Scalar SizeX() const { return size_x_; }
    inline Scalar SizeY() const { return size_y_; }
  
    // Conic Project(const Circle& circle) const;
    Vector2 Project(const Vector3& point) const;
    Line2d Project(const Circle& circle) const;
    Ellipse Project(const Sphere& sphere) const;
    Conic ProjectCircle(const Circle& circle) const;
    // std::pair<cv::Point2f, cv::Point2f> Project(const Circle& circle, Scalar length) const;
  
    DualCircle Unproject(const Ellipse& ellipse, const Scalar radius) const;
    Vector3 Unproject(const Vector2& point2d, const Scalar distance) const;
  private:
    Scalar focal_length_;
    Scalar cx_, cy_;
    Scalar size_x_, size_y_;
};

inline int sign(Scalar x) {
  if (x > 0) return 1;
  else if (x < 0) return -1;
  else return 0;
}

}  // namespace pupil

#endif  // _SRC_CAMERA_H_