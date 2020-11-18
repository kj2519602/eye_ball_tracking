#ifndef _PUPIL_DETECTOR_H_
#define _PUPIL_DETECTOR_H_

#include "opencv2/opencv.hpp"
#include "geometry.h"

namespace pupil {

class PupilDetector {
public:
  using PixelType = uint8_t;
  struct Options {
      // 1. processing
      size_t crop_margin = 20;
      size_t blur_size = 5;
      // 2. dark_region_roi
      size_t init_roi_size = 140;
      size_t down_sample_factor = 4;
      // 3. image thresh
      size_t dark_bin_size = 50;
      PixelType thresh_margin = 10;
      size_t mask_margin = 10;
      // 4. Canny edge detector
      size_t low_threshold = 10;
      size_t ratio = 3; 
      int kernel_size = 3;
      bool debug = false;
  };

  explicit PupilDetector(const Options& options) : options_(options) {};
  Ellipse Detect(const cv::Mat& frame);

private:
  // Get square roi with darkest mean pixel value.
  cv::Rect GetDarkestRoi(const cv::Mat& input);

  // Dark Threshold = Mean(K darkest pixel value) + thresh_margin.
  const PixelType GetDarkThreshold(const cv::Mat& input);

  // Get BW pupil mask region. 
  size_t GetPupilMask(const cv::Mat& input, const PixelType thresh, cv::Mat* mask);

  // Get Mask ROI.
  cv::Rect GetMaskRoi(const cv::Mat& mask);

  // Get pupil edges under pupil mask with Canny detector.
  void GetPupilEdges(const cv::Mat& input, const cv::Mat& mask, cv::Mat* edges, std::vector<cv::Point>& corner_points);

  // 
  void FindBestEdges();

  void FitPupilElipse();

  const Options options_;
};

}  // namespace pupil

#endif  // _PUPIL_DETECTOR_H_