#ifndef CAT_DETECTION_RESNET_H
#define CAT_DETECTION_RESNET_H

#include <opencv2/opencv.hpp>


#include <torch/torch.h>
#include <torch/script.h>

#include <vector>

namespace resnet {

    class ResNet {
    public:
        explicit ResNet(const std::string& _file_path);

        torch::Tensor inference(const cv::Mat& _inp_mat);

        inline bool isLoad() const{
            return this->is_load;
        }

        ~ResNet() = default;

    private:
        cv::Mat imgProcess(const cv::Mat &_inp_mat);

        torch::Tensor cvMat2Tensor(const cv::Mat &_inp_mat);

    private:
        torch::jit::script::Module module;

        cv::Size size_ = cv::Size(224,224);

        bool is_load = false;
    };

} // resnet

#endif //CAT_DETECTION_RESNET_H
