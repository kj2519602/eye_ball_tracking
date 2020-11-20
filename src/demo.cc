#include "eyeball_tracker.h"

#include <string>
#include <iostream>
#include <vector>
#include <algorithm>    // std::sort
#include <unordered_map>

#include "geometry.h"
#include "camera.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
// #include "opencv2/core/utils/filesystem.hpp"


int get_frame_id(const std::string& file_path) {
	const std::size_t dot_pos = file_path.rfind('.');
	const std::size_t sep_pos = file_path.rfind('_');
	return std::stoi(file_path.substr(sep_pos + 1, file_path.size() - dot_pos));
}

std::size_t get_ordered_frames(const std::string& input_path, std::vector<cv::String>& frame_files) {
    cv::glob(input_path + "\\*.tiff", frame_files, false); 
    std::sort(frame_files.begin(), frame_files.end(),
                [](const std::string& a, const std::string& b){return get_frame_id(a) < get_frame_id(b);});
    return frame_files.size();
}

void run(const std::string& input_path, int fps = 50) {
    std::vector<cv::String> frame_files;
    auto frame_size = get_ordered_frames(input_path, frame_files);
    size_t width = 320;
    size_t height = 240;

    double focal_length = 1.06079493e+03;
    pupil::Camera camera(focal_length, width/2, height/2, width, height);

    pupil::EyeballTracker::Options options;
    pupil::EyeballTracker pupil_tracker(camera, options);

    cv::namedWindow("Pupil", cv::WINDOW_NORMAL);
    cv::namedWindow("Debug_1", cv::WINDOW_NORMAL);
    std::vector<pupil::Line2d> lines;
    int index = 0;
    while(1) {
        const std::string filename = frame_files[index];
        index = (index+1) % frame_size;
        cv::Mat frame = cv::imread(filename);
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        const auto eye = pupil_tracker.Track(gray);

        if (pupil_tracker.IsInitialized() && eye.is_valid) {
            auto rect = eye.pupil.Rect();
            // Draw pupil circle on image.
            cv::ellipse(frame, rect, cv::Scalar(0, 255, 255), 1);
            // Draw eye ball.
            cv::ellipse(frame, eye.eye_ball.Rect(), cv::Scalar(255, 255, 0), 1);
            // Draw pupil center.
            auto pupil = camera.Project(eye.circle.position);
            cv::Point2f pupil_center(pupil[0], pupil[1]);
            cv::circle(frame, pupil_center, 2, cv::Scalar(0, 0, 255));
            // Draw eye ball center.
            auto center = eye.eye_ball.Center();
            cv::Point2f eyeball_center(center[0], center[1]);
            cv::circle(frame, eyeball_center, 5, cv::Scalar(0, 255, 0));
            // Draw gaze vector.
            auto gaze_end = camera.Project(eye.circle.position + (eye.circle.norm * 20)); 
            cv::line(frame, pupil_center, cv::Point2f(gaze_end[0], gaze_end[1]),
                cv::Scalar(0, 0, 255), 1);
            cv::line(frame, eyeball_center, pupil_center, cv::Scalar(0, 255, 0), 1);
            // Log angular results.
            std::cout << "------------------------------" << std::endl;
            std::cout << "Theta: " << eye.params.theta << std::endl;
            std::cout << "Psi: " << eye.params.psi << std::endl;
        }
        cv::imshow("Pupil", frame);
        if(cv::waitKey(1000/fps) == 12) { 
            break; 
        }
    }
    
}

int main(int argc, char *argv[]){
    std::string input_path = argv[1]; // input path.
    run(input_path);
    return 0;
}