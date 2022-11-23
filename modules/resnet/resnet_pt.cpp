//
// Created by rcxxx on 22-10-28.
//

#include "resnet_pt.h"

namespace resnet {

    ResNet::ResNet(const std::string& _file_path) {
        try {
            module = torch::jit::load(_file_path);
            is_load = true;
        }
        catch (const c10::Error& ){
            std::cerr << "error loading the model\n";
            is_load = false;
        }
    }

    torch::Tensor ResNet::inference(const cv::Mat &_inp_mat) {
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(this->cvMat2Tensor(_inp_mat));
        torch::Tensor out = this->module.forward(inputs).toTensor();

        out = torch::flatten(out, 1);
        out = torch::nn::functional::normalize(out, torch::nn::functional::NormalizeFuncOptions().dim(1));

        return out;
    }

    cv::Mat ResNet::imgProcess(const cv::Mat &_inp_mat) {
        cv::Mat dst_mat;
        cv::cvtColor(_inp_mat, dst_mat, cv::COLOR_BGR2RGB);
        cv::resize(dst_mat, dst_mat , this->size_);

        //normalization
        dst_mat.convertTo(dst_mat, CV_64FC3, 1.0f / 255.0f);
        cv::Mat mean = cv::Mat(dst_mat.size(), dst_mat.type(), cv::Scalar_<double>(0.485, 0.456, 0.406));
        dst_mat -= mean;
        cv::Mat std = cv::Mat(dst_mat.size(), dst_mat.type(), cv::Scalar_<double>(0.229, 0.224, 0.225));
        dst_mat /= std;

        return dst_mat;
    }

    torch::Tensor ResNet::cvMat2Tensor(const cv::Mat &_inp_mat) {
        cv::Mat inp_rgb = this->imgProcess(_inp_mat);
        //opencv format H*W*C
        auto out_tensor = torch::from_blob(inp_rgb.data, {1, inp_rgb.rows, inp_rgb.cols, 3}, torch::kByte);
        //pytorch format N*C*H*W
        out_tensor = out_tensor.permute({0, 3, 1, 2});

        return out_tensor.toType(torch::kFloat);
    }
} // resnet