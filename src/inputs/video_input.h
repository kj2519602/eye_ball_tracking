#ifndef _SRC_INPUTS_VIDEO_INPUT_H_
#define _SRC_INPUTS_VIDEO_INPUT_H_

#include <string>

#include "inputs/abstract_input.h"

#include "opencv2/opencv.hpp"

namespace pupil {

class VideoInput : public AbstractInput {
  public:
    VideoInput(const std::string& filename);
    ~VideoInput() override;

    bool GrabFrame(cv::Mat* frame) override;
  private:
    cv::VideoCapture cap_;
};

}  // namespace pupil

#endif//_SRC_INPUTS_VIDEO_INPUT_H_