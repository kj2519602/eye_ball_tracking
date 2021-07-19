#include "core/eyeball_tracker.h"

#include <string>
#include <iostream>
#include <vector>
#include <algorithm>    // std::sort
#include <unordered_map>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "math/geometry.h"
#include "math/camera.h"
#include "inputs/video_input.h"

namespace pupil {

class DemoApp {
  public:
    struct DemoAppOptions {
        std::string filename;
        double focal_length = 946.0; // 1.06079493e+03;
        pupil::EyeballTracker::Options tracker_options;
        bool visualize = true;
    };

    DemoApp(const DemoAppOptions& options);

    bool Run();
    bool RunOnce();
    void Visualize();
  private:
    std::unique_ptr<AbstractInput> input_;
    cv::Mat frame_;
    bool visualize_;
    EyeballTracker::Eyeball eye_;
    pupil::Camera camera_;
    pupil::EyeballTracker pupil_tracker_;
};

DemoApp::DemoApp(const DemoAppOptions& options) :
input_(new VideoInput(options.filename)),
camera_(options.focal_length, input_->Width()/2, input_->Height()/2, input_->Width(), input_->Height()),
pupil_tracker_(camera_, options.tracker_options),
visualize_(options.visualize) {
} 

bool DemoApp::Run() {
    while (input_->HasNext()) {
        RunOnce();
        if (visualize_) Visualize();
    }
    return true;
}

bool DemoApp::RunOnce() {
    input_->GrabFrame(&frame_);
    if (frame_.channels() == 3) {
        cv::cvtColor(frame_, frame_, cv::COLOR_BGR2GRAY);
    }
    eye_ = pupil_tracker_.Track(frame_);
    return eye_.is_valid;
}

void DemoApp::Visualize() {
    cv::Mat frame_draw;
    cv::cvtColor(frame_, frame_draw, cv::COLOR_GRAY2BGR);
    // Draw pupil circle on image.
    auto rect = eye_.pupil.Rect();
    cv::ellipse(frame_draw, rect, cv::Scalar(0, 255, 255), 1);
    if (pupil_tracker_.IsInitialized() && eye_.is_valid) {
        // Draw rris circle on image.
        auto circle = eye_.circle;
        circle.radius *= 5e-2;
        Ellipse iris(camera_.ProjectCircle(circle));
        iris.cx += camera_.CenterX();
        iris.cy += camera_.CenterY();
        std::cout << "iris center: " << iris.cx << " " << iris.cy << std::endl;
        std::cout << "iris radius: " << iris.major_radius << std::endl;
        cv::ellipse(frame_draw, iris.Rect(), cv::Scalar(0, 0, 255), 1);

        // Draw eye ball.
        cv::ellipse(frame_draw, eye_.eye_ball.Rect(), cv::Scalar(255, 255, 0), 1);

        // Draw pupil center.
        auto pupil = camera_.Project(eye_.circle.position);
        cv::Point2f pupil_center(pupil[0], pupil[1]);
        cv::circle(frame_draw, pupil_center, 2, cv::Scalar(0, 0, 255));

        // Draw eye ball center.
        auto center = eye_.eye_ball.Center();
        cv::Point2f eyeball_center(center[0], center[1]);
        cv::circle(frame_draw, eyeball_center, 5, cv::Scalar(0, 255, 0));
        // Draw gaze vector.
        auto gaze_end = camera_.Project(eye_.circle.position + (eye_.circle.norm * 20)); 
        cv::line(frame_draw, pupil_center, cv::Point2f(gaze_end[0], gaze_end[1]),
            cv::Scalar(0, 0, 255), 1);
        cv::line(frame_draw, eyeball_center, pupil_center, cv::Scalar(0, 255, 0), 1);

        // Log angular results.
        std::cout << "------------------------------" << std::endl;
        std::cout << "Theta: " << eye_.params.theta << std::endl;
        std::cout << "Psi: " << eye_.params.psi << std::endl;
    }
    cv::imshow("Pupil", frame_draw);
    cv::waitKey(1000 / input_->Fps());
}

}  // namespace pupil

int main(int argc, char *argv[]){
    const std::string input_path = argv[1]; // input path.
    pupil::DemoApp::DemoAppOptions options;
    options.filename = input_path;

    auto demo_app = pupil::DemoApp(options);
    demo_app.Run();
    return 0;
}