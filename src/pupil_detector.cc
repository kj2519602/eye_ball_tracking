#include "pupil_detector.h"

#include <vector>
#include <iostream>
#include <limits>
#include <algorithm>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

namespace pupil {
// Run entire pupil detector algorithm.
Ellipse PupilDetector::Detect(const cv::Mat& frame) {
    cv::Rect crop_rect = cv::Rect(options_.crop_margin,
                                  options_.crop_margin, 
                                  frame.cols - 2*options_.crop_margin, 
                                  frame.rows - 2*options_.crop_margin);

    cv::Mat smoothed = frame(crop_rect);
    cv::normalize(smoothed, smoothed, 255, 0, cv::NORM_MINMAX, CV_8UC1);
    cv::GaussianBlur(smoothed, smoothed, cv::Size(options_.blur_size, options_.blur_size), 0, 0);
    cv::Rect roi_rect =  GetDarkestRoi(smoothed);

    cv::Mat roi = smoothed(roi_rect);
    auto dark_thresh = GetDarkThreshold(roi);
    cv::Mat mask;
    auto mask_size = GetPupilMask(roi, dark_thresh, &mask);
    cv::Rect roi_roi_rect = GetMaskRoi(mask);
    mask = mask(roi_roi_rect);
    roi = roi(roi_roi_rect);
    roi_rect = cv::Rect(roi_rect.x + roi_roi_rect.x,
                        roi_rect.y + roi_roi_rect.y, 
                        roi_roi_rect.width, roi_roi_rect.height);

    cv::Mat pupil_edges;
    std::vector<cv::Point> counters;
    GetPupilEdges(roi, mask, &pupil_edges, counters);
    for (auto& point : counters) {
        point.x += roi_rect.x + options_.crop_margin;
        point.y += roi_rect.y + options_.crop_margin;
    }
    auto rect = cv::fitEllipse(counters);
    assert(rect.size.width <= rect.size.height);
    return Ellipse(rect);
}

void correctRoi(const int width, const int height, cv::Rect* roi_rect) {
    roi_rect->x = std::max(roi_rect->x, 0);
    roi_rect->y = std::max(roi_rect->y, 0);
    roi_rect->width = std::min(roi_rect->width, width - (roi_rect->x) - 1);
    roi_rect->height = std::min(roi_rect->height, height - (roi_rect->y) - 1);
}

double min_pixel_value(const cv::Mat& input) {
    double min_value;
    cv::minMaxLoc(input, &min_value, NULL, NULL);
    return min_value;
}

double max_pixel_value(const cv::Mat& input) {
    double max_value;
    cv::minMaxLoc(input, NULL, &max_value, NULL);
    return max_value;
}

// Get square roi with darkest mean pixel value.
cv::Rect PupilDetector::GetDarkestRoi(const cv::Mat& input) {
    cv::Mat down_sampled;
    auto df = options_.down_sample_factor;
    cv::resize(input, down_sampled, cv::Size(input.cols/df, input.rows/df), 0, 0, cv::INTER_NEAREST);
    cv::Point darkest_point;
    cv::minMaxLoc(down_sampled, NULL, NULL, &darkest_point);
    cv::Rect roi_rect = cv::Rect(darkest_point.x * df - options_.init_roi_size/2,
                                 darkest_point.y * df - options_.init_roi_size/2, 
                                 options_.init_roi_size, 
                                 options_.init_roi_size);
    correctRoi(input.cols, input.rows, &roi_rect);
    return roi_rect;
}

// Dark Threshold = Mean(K darkest pixel value) + thresh_margin.
const PupilDetector::PixelType PupilDetector::GetDarkThreshold(const cv::Mat& input) {
    std::vector<PixelType> darkest_k_pixels;
    for( int i=0; i<input.rows; ++i){
        const PixelType* ptr = input.ptr<const PixelType>(i);
        for( int j=0; j<input.cols;++j){
            darkest_k_pixels.push_back(ptr[j]);
            std::push_heap(darkest_k_pixels.begin(),darkest_k_pixels.end());
            if (darkest_k_pixels.size() > options_.dark_bin_size) {
                std::pop_heap(darkest_k_pixels.begin(),darkest_k_pixels.end());
                darkest_k_pixels.pop_back();
            }
        }
    }
    std::make_heap(darkest_k_pixels.begin(),darkest_k_pixels.end());
    PixelType thresh = darkest_k_pixels.front();

    //std::accumulate(darkest_k_pixels.begin(), darkest_k_pixels.end(), 0.0)/darkest_k_pixels.size();
    return thresh + options_.thresh_margin;
}

// Get BW pupil mask region. 
size_t PupilDetector::GetPupilMask(const cv::Mat& input, const PixelType thresh, cv::Mat* mask) {
    cv::threshold(input, *mask, thresh, 1, cv::THRESH_BINARY);
    cv::Mat dist, labels;
    cv::distanceTransform(*mask, dist, labels, 3,  cv::DIST_L1);
    cv::threshold(dist, *mask, options_.mask_margin, 65536, cv::THRESH_BINARY_INV);
    mask->convertTo(*mask,CV_8UC1);
    return 0;
}

// Get Mask ROI.
cv::Rect PupilDetector::GetMaskRoi(const cv::Mat& mask) {
    cv::Mat Points;
    cv::findNonZero(mask, Points);
    return cv::boundingRect(Points);
}

// Get pupil edges under pupil mask with Canny detector.
void PupilDetector::GetPupilEdges(const cv::Mat& input, const cv::Mat& mask, cv::Mat* edges, std::vector<cv::Point>& corner_points) {
    cv::Mat labels; 
    cv::Canny(input, *edges, options_.low_threshold, options_.low_threshold*options_.ratio, options_.kernel_size);
    edges->copyTo(*edges, mask);
    int nLabels = cv::connectedComponents(*edges, labels, 8);
    std::vector<size_t> counter(nLabels, 0);
    for(int r = 0; r < labels.rows; ++r){
        for(int c = 0; c < labels.cols; ++c){
            int label = labels.at<int>(r, c);
            if (label != 0)
                counter[label]++;
         }
    }
    const size_t max_label = std::max_element(counter.begin(), counter.end()) - counter.begin();
    cv::Mat filtered_edges(edges->size(), CV_32FC1, cv::Scalar(0));
    for(int r = 0; r < labels.rows; ++r){
        for(int c = 0; c < labels.cols; ++c){
            int label = labels.at<int>(r, c);
            if (label == max_label){
                corner_points.push_back(cv::Point(c, r));
                filtered_edges.at<float>(r, c) = 1;
            }
         }
    }
}

// 
void FindBestEdges();

}   // namespace pupil