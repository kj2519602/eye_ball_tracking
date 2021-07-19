#ifndef _SRC_INPUTS_ABSTRACT_INPUT_H_
#define _SRC_INPUTS_ABSTRACT_INPUT_H_

#include "opencv2/opencv.hpp"

namespace pupil {

class AbstractInput {
  public:
    AbstractInput() = default;
    virtual ~AbstractInput() = default;

    virtual bool GrabFrame(cv::Mat* frame) = 0;

    int Width() const { return width_; }
    int Height() const { return height_; }
    float Fps() { return fps_; }
    bool HasNext() { return has_next_; }
  protected:
    int width_ = 0;
    int height_ = 0;
    float fps_;
    bool has_next_ = false;
};

std::unique_ptr<AbstractInput> CreateInput();

}  // namespace pupil

#endif//_SRC_INPUTS_ABSTRACT_INPUT_H_