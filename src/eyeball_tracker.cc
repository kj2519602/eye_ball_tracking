#include "eyeball_tracker.h"
#include "intersect.h"

namespace pupil {

EyeballTracker::Eyeball EyeballTracker::Track(const cv::Mat& frame) {
    // Detect pupil ellipse.
    eye_.pupil = pupil_detector_->Detect(frame);
    if (!IsInitialized()) {
        Init(eye_.pupil);
    } else {
        Run(eye_.pupil);
    }
    return eye_;
}

void EyeballTracker::Init(const Ellipse& pupil) {
    // Unproject pupil.pupil_radius.
    int gx = pupil.cx / grid_sz_;
    int gy = pupil.cy / grid_sz_;
    int idx = gy * camera_.SizeX() / grid_sz_ + gx;
    if (!occupation_bins_[idx]) {
        const auto dual_circle = camera_.Unproject(pupil, options_.pupil_radius);
        occupation_bins_[idx] = true;
        observations_.push_back(dual_circle);
    }
    if (observations_.size() == options_.init_frame_num) {
        EstimateEyeSphere();
        observations_.clear();
        Run(pupil);
        is_initialized_ = true;
    }
}

bool EyeballTracker::Run(const Ellipse& pupil) {
    // Unproject pupil.pupil_radius.
    const auto dual_circle = camera_.Unproject(pupil, options_.pupil_radius);
    eye_.circle = GetValidPupilCircle(dual_circle, eye_.eye_ball.Center());
    try {
        auto line_pupil = Line3d(Vector3::Zero(), eye_.circle.position.normalized());
        auto point_pair = intersect_line_sphere(line_pupil, eye_.sphere);
        auto result = (point_pair.first[2] < point_pair.second[2]) ? point_pair.first : point_pair.second;
        eye_.circle.position = point_pair.first;
        Vector3 eye_center_to_pupil = eye_.circle.position - eye_.sphere.center;
        eye_.circle.norm = eye_center_to_pupil.normalized();
        // Assign eye params.
        eye_.params.theta = std::acos(eye_center_to_pupil[1] / eye_center_to_pupil.norm());
        eye_.params.psi = std::atan2(eye_center_to_pupil[2], eye_center_to_pupil[0]);
        eye_.is_valid = true;
        return true;
    } catch (no_intersection_exception&) {
        std::cout << "No intersection!" << std::endl;
        eye_.is_valid = false;
        return false;
    }
}

void EyeballTracker::EstimateEyeSphere() {
    // Find eyeball center projection on image through line intersection.
    std::vector<Line2d> lines;
    lines.reserve(observations_.size());
    for (const auto& obs : observations_) {
        lines.push_back(camera_.Project(obs.first));
    } 
    // Find eyeball center through unpojection.
    eye_.sphere.center = camera_.Unproject(intersect_lines(lines), options_.eye_distance);
     // Find eyeball radius.
    Scalar eye_radius_acc = 0.0;
    size_t eye_radius_count = 0;
    Scalar r = 0.0;
    for (const auto& obs : observations_) {
        const auto pupil_circle = GetValidPupilCircle(obs, eye_.eye_ball.Center());
        const auto pupil_position = RefinePupilPosition(eye_.sphere.center, pupil_circle);
        auto r_cur = (pupil_position - eye_.sphere.center).norm();
        if (r_cur > r) r = r_cur;
        eye_radius_acc += (pupil_position - eye_.sphere.center).norm();
        eye_radius_count++;
    }
    // Calculate mean.
    assert(eye_radius_count > 0);
    eye_.sphere.radius = r; // eye_radius_acc / eye_radius_count;
    // Adjust to real eye ball size;
    const Scalar true_radius = 12.0;
    auto scale = true_radius / eye_.sphere.radius;
    eye_.sphere.radius = true_radius;
    eye_.sphere.center *= scale;
    eye_.eye_ball = camera_.Project(eye_.sphere);
}

Circle EyeballTracker::GetValidPupilCircle(const DualCircle& dual_circle,
                                           const Vector2& eye_center) const {
    if (IsValidPupilCircle(dual_circle.first, eye_center)) {
        return dual_circle.first;
    }
    assert(IsValidPupilCircle(dual_circle.second, eye_center));
    return dual_circle.second;
}

bool EyeballTracker::IsValidPupilCircle(const Circle& circle, const Vector2& eye_center) const {
    const auto line = camera_.Project(circle);
    return (line.direction().dot(line.origin() - eye_center) > 0.0);
}

Vector3 EyeballTracker::RefinePupilPosition(const Vector3& eye_center,
                                       const Circle& circle_in) const {
    const std::vector<Line3d> lines = {Line3d(eye_center, circle_in.norm),
                                       Line3d(Vector3::Zero(), circle_in.position.normalized())};
    return intersect_lines(lines);
}

}  // namespace pupil