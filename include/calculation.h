#ifndef CAT_RECOGNITION_CALCULATION_H
#define CAT_RECOGNITION_CALCULATION_H

#include <opencv2/opencv.hpp>

#include <torch/torch.h>

static cv::Mat rmRoiBoundary(const cv::Mat& _src_img, const cv::Rect& _bbox, bool _rm_boundary = false, float scale = 0.2){
    cv::Rect bbox = _bbox;

    // Over Bounds
    if (bbox.tl().x < 0){
        bbox.width += bbox.tl().x;
        bbox.tl() = cv::Point(0, bbox.tl().y);
    }
    if (bbox.tl().y < 0){
        bbox.height += bbox.tl().y;
        bbox.tl() = cv::Point(bbox.tl().x, 0);
    }
    if (bbox.tl().x + bbox.width > _src_img.cols) {
        bbox.width = _src_img.cols - bbox.tl().x;
    }
    if (bbox.tl().y + bbox.height > _src_img.rows) {
        bbox.height = _src_img.rows - bbox.tl().y;
    }

    // crop && zoom
    if (_rm_boundary) {
        int scale_w = static_cast<int>(static_cast<float>(bbox.width)  * scale);
        int scale_h = static_cast<int>(static_cast<float>(bbox.height) * scale);
        cv::Rect bbox_zoom = bbox - cv::Size(scale_w, scale_h);
        cv::Point pt;
        pt.x = cvRound(scale_w/2.0);
        pt.y = cvRound(scale_h/2.0);
        bbox_zoom += pt;
        return _src_img(bbox_zoom);
    } else {
        return _src_img(bbox);
    }
}

static std::vector<float> tensorEuclideanDistance(const torch::Tensor& t_1, const torch::Tensor& t_2){
    auto euclidean_dist = torch::nn::functional::pairwise_distance(t_1,
                                                                   t_2,
                                                                   torch::nn::functional::PairwiseDistanceFuncOptions().p(2));
    std::vector<float> dist(euclidean_dist.data_ptr<float>(), euclidean_dist.data_ptr<float>() + euclidean_dist.numel());

    return dist;
}

#endif //CAT_RECOGNITION_CALCULATION_H
