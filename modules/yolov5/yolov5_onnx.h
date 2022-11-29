#ifndef YOLOV5_CAT_DET_YOLOV5_ONNX_H
#define YOLOV5_CAT_DET_YOLOV5_ONNX_H

#include <fstream>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace yolov5 {

    struct Detection
    {
        int class_id;
        float confidence;
        cv::Rect bbox;
    };


    class Net {
    public:
        /**
         * @brief Construct a new Net object
         *
         * @param onnx_model_path yolov5 onnx models file path
         * @param class_list_path class_list file path
         * @param is_cuda is CUDA supported
         */
        Net(const std::string& onnx_model_path,
            const std::string& class_list_path,
            float resolution_  = 480,
            bool is_cuda  = false);
        /**
         * @brief Destroy the Net object
         *
         */
        ~Net()= default;

        /**
         * @brief
         *
         * @param src
         * @param _score_threshold
         * @param _NMS_threshold
         * @param _confidence_threshold
         * @return std::vector<yolov5_onnx::Detection>
         */
        std::vector<yolov5::Detection> detect(cv::Mat &src,
                                              float _score_threshold = 0.2,
                                              float _nms_threshold = 0.4,
                                              float _confidence_threshold = 0.4);



        inline std::vector<std::string> classList() const{
            return this->class_list;
        }

    private:
        cv::dnn::Net net;
        std::vector<std::string> class_list;

        float resolution;
        int output_dimensions;
        int output_rows;

        static cv::Mat format_img(const cv::Mat &src);
    };

} // yolov5

#endif //YOLOV5_CAT_DET_YOLOV5_ONNX_H
