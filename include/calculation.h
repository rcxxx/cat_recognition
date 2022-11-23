//
// Created by rcxxx on 22-11-22.
//

#ifndef CAT_RECOGNITION_CALCULATION_H
#define CAT_RECOGNITION_CALCULATION_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

cv::Mat rmBoundary(const cv::Mat& _src_img, const cv::Rect& _bbox, bool _rm_boundary = false, float scale = 0.2){
    if (_rm_boundary) {
        int scale_w = static_cast<int>(_bbox.width  * scale);
        int scale_h = static_cast<int>(_bbox.height * scale);
        cv::Rect bbox_zoom = _bbox - cv::Size(scale_w, scale_h);
        cv::Point pt;
        pt.x = cvRound(scale_w/2.0);
        pt.y = cvRound(scale_h/2.0);
        bbox_zoom -= pt;
        return _src_img(bbox_zoom);
    } else {
        return _src_img(_bbox);
    }
}

std::vector<float> tensorEuclideanDistance(const torch::Tensor& t_1, const torch::Tensor& t_2){
    auto euclidean_dist = torch::nn::functional::pairwise_distance(t_1,
                                                                   t_2,
                                                                   torch::nn::functional::PairwiseDistanceFuncOptions().p(2));
    std::vector<float> dist(euclidean_dist.data_ptr<float>(), euclidean_dist.data_ptr<float>() + euclidean_dist.numel());

    return dist;
}

#endif //CAT_RECOGNITION_CALCULATION_H
