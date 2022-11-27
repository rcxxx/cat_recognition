#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "yolov5/yolov5_onnx.h"
#include "resnet/resnet_pt.h"
#include "calculation.h"
#include "local_feature.h"


int main() {
    // Param Init
    cv::FileStorage fs("../config.yml", cv::FileStorage::READ);
    std::string yolo_model_path;
    fs["yolo_model_path"]   >> yolo_model_path;
    std::string classes_path;
    fs["classes_path"]      >> classes_path;
    std::string resnet50_model_path;
    fs["resnet_model_path"] >> resnet50_model_path;
    int device_num;
    fs["device"]            >> device_num;
    std::string video_path;
    fs["video_path"] >> video_path;
    std::string feature_csv_path;
    fs["feature_csv_path"]  >> feature_csv_path;

    fs.release();

    // Load Model
    cv::Scalar      box_color = cv::Scalar(0, 255, 255);
    yolov5::Net     yolo(yolo_model_path, classes_path);
    resnet::ResNet  resnet50(resnet50_model_path);
    if (!resnet50.isLoad()){
        return -1;
    }

    // Load Local Feature List
    std::ifstream read_handel(feature_csv_path, std::ios_base::in);
    /* <-- feat_list --> */
    std::vector<local_feature> feat_list = loadFeature(read_handel);
    /* <-- feat_list --> */
    read_handel.close();

    // Capture Stream
    cv::VideoCapture cap;
    cap.open(device_num);
    if(!cap.isOpened()){
        cap.open(video_path);
        if (!cap.isOpened()){
            std::cout << "Can't open video stream" << std::endl;
            return -1;
        }
        if (cap.get(cv::CAP_PROP_FRAME_WIDTH)> cap.get(cv::CAP_PROP_FRAME_HEIGHT)){
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 720);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        }
        else
        {
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 480);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        }
    }
    else{
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 720);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    }

    /**-    Loop    -**/
    cv::Mat src_img;
    while (char(cv::waitKey(1)) != 27) {
        cap >> src_img;

        // YOLO detect
        std::vector<yolov5::Detection> result =  yolo.detect(src_img);

        for (const auto &idx : result){
            if (idx.class_id == 15 || idx.class_id == 16){
                auto bbox = idx.bbox;
                cv::rectangle(src_img, bbox, box_color, 2);
                cv::rectangle(src_img, cv::Point(bbox.x, bbox.y + 10), cv::Point(bbox.x + bbox.width, bbox.y), box_color, cv::FILLED);
                cv::putText(src_img, yolo.classList()[15], cv::Point(bbox.x, bbox.y + 5), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);

                cv::Mat crop_img = rmRoiBoundary(src_img, bbox, true, 0.2);
                torch::Tensor feature_self = resnet50.inference(crop_img);
            }
        }
    }
    /**-    Loop    -**/

    cv::destroyAllWindows();
    return 0;
}
