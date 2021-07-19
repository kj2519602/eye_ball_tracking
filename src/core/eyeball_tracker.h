#ifndef _SRC_CORE_EYEBALL_TRACKER_H_
#define _SRC_CORE_EYEBALL_TRACKER_H_

#include <string.h>
#include <memory>
#include <vector>

#include "core/pupil_detector.h"
#include "math/geometry.h"
#include "math/camera.h"

namespace pupil {

class EyeballTracker {
  public:
    struct Options {
        PupilDetector::Options detector_option;
        size_t init_frame_num = 50;
        Scalar pupil_radius = 100.0;
        Scalar eye_distance = 100.0;
        int grid_sz = 8;
    };

    struct Params {
        double theta = 0.0;
        double psi = 0.0;
    };

    struct Eyeball {
        Ellipse pupil;
        Ellipse eye_ball;
        Circle circle;
        Sphere sphere;
        bool is_valid;
        Params params;
        Eyeball() = default;
        Eyeball(const Ellipse& ellipse, const Circle& circle, const Sphere& sphere);
    };

    EyeballTracker(const Camera& camera, const Options& options)
        : eye_(Eyeball()),
        is_initialized_(false),
        grid_sz_(options.grid_sz),
        observations_(std::vector<DualCircle> ()), camera_(camera),
        pupil_detector_(new PupilDetector(options.detector_option)),
        options_(options) {
        const auto bin_sz = camera_.SizeX() * camera_.SizeY() / (grid_sz_ * grid_sz_);
        occupation_bins_.resize(bin_sz, false);
    }

    Eyeball Track(const cv::Mat& frame);
    inline bool IsInitialized() const {
        return is_initialized_;
    };

private:
    void Init(const Ellipse& pupil);
    bool Run(const Ellipse& pupil);
    void EstimateEyeSphere();
    void EstimateGaze(const DualCircle& dual_circle);
    Circle GetValidPupilCircle(const DualCircle& dual_circle, const Vector2& eye_center) const;
    Vector3 RefinePupilPosition(const Vector3& eye_center, const Circle& circle_in) const;
    bool IsValidPupilCircle(const Circle& circle, const Vector2& eye_center) const;


    Eyeball eye_;
    bool is_initialized_;
    int grid_sz_;
    std::vector<bool> occupation_bins_;
    std::vector<DualCircle> observations_;
    Camera camera_;
    std::unique_ptr<PupilDetector> pupil_detector_;
    Options options_;
};


}  // namespace pupil

#endif  // _SRC_CORE_EYEBALL_TRACKER_H_