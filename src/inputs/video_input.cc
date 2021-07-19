#include "inputs/video_input.h"

namespace pupil {

VideoInput::VideoInput(const std::string& filename) : cap_(cv::VideoCapture(filename)) {
    if (cap_.isOpened()) {
        has_next_ = true;
        width_ = cap_.get(cv::CAP_PROP_FRAME_WIDTH); 
        height_ = cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
        fps_ = cap_.get(cv::CAP_PROP_FPS); 
    }
}

VideoInput::~VideoInput() {
    cap_.release();
}

// Must be called after HasNext().
bool VideoInput::GrabFrame(cv::Mat* frame) {
    const bool success = cap_.read(*frame); // read a new frame from video.
    has_next_ = cap_.isOpened() && success;
    return success;
}

}  // namespace pupil