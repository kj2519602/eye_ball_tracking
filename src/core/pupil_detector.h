#ifndef _SRC_CORE_PUPIL_DETECTOR_H_
#define _SRC_CORE_PUPIL_DETECTOR_H_

#include "opencv2/opencv.hpp"
#include "math/geometry.h"

namespace pupil {

class PupilDetector {
public:
  using PixelType = uint8_t;

  struct Options {
      // 1. processing
      size_t crop_x_left = 100;
      size_t crop_x_right = 200;
      size_t crop_y_top = 30;
      size_t crop_y_bottom = 60;
      size_t blur_size = 5;
      // 2. dark_region_roi
      size_t init_roi_size = 200;
      size_t down_sample_factor = 8;
      // 3. image thresh
      size_t dark_bin_size = 50;
      PixelType thresh_margin = 10;
      size_t mask_margin = 5;
      // 4. Canny edge detector
      size_t low_threshold = 10;
      size_t ratio = 3; 
      int kernel_size = 3;
      bool debug = false;
      // 5. Inlier filter.
      float ellipse_inlier_min_dist_0 = 2.5;
      float ellipse_inlier_min_dist_1 = 1.0;
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
  size_t GetPupilEdges(const cv::Mat& input, const cv::Mat& mask, cv::Mat* edges, std::vector<cv::Point>* corner_points);

  size_t GetEllipseInliers(const Ellipse& ellipse, const std::vector<cv::Point>& raw_edges, float min_dist, std::vector<cv::Point>* inliers);

  const Options options_;
};

size_t MaxConnectedComponent(cv::Mat* binary);

size_t ConnectedComponents(const cv::Mat& input, cv::Mat* output);

}  // namespace pupil

#endif  // _SRC_CORE_PUPIL_DETECTOR_H_