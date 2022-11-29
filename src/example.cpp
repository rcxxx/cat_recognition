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

    std::vector<cv::Mat> images;
    images.push_back(cv::imread("../img/gold.jpeg"));
    images.push_back(cv::imread("../img/blue.jpeg"));

    for (auto & img:images) {
        cv::resize(img, img, cv::Size(720, 480));
        // YOLO detect
        std::vector<yolov5::Detection> result =  yolo.detect(img);

        for (const auto &idx : result){
            if (idx.class_id == 15 || idx.class_id == 16){
                auto bbox = idx.bbox;
                cv::rectangle(img, bbox, box_color, 2);
                cv::rectangle(img, cv::Point(bbox.x, bbox.y + 10), cv::Point(bbox.x + bbox.width, bbox.y-10), box_color, cv::FILLED);
                cv::putText(img, yolo.classList()[15], cv::Point(bbox.x, bbox.y + 5), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 1.2);

                cv::Mat crop_img = rmRoiBoundary(img, bbox, true, 0.2);
                torch::Tensor feature_self = resnet50.inference(crop_img);

                float min_dist = FLT_MAX;
                std::string crop_id;

                for (const auto& f:feat_list){
                    torch::Tensor feature_local = torch::tensor(f.data_row);
                    float dist_tmp = tensorEuclideanDistance(feature_self, feature_local)[0];
                    std::cout << dist_tmp << " " <<std::endl;
                    if (dist_tmp <= min_dist){
                        min_dist = dist_tmp;
                        crop_id = f.id;
                    }
                }

                cv::putText(img, crop_id, cv::Point(bbox.x + bbox.width*0.5, bbox.y + 5), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 1.2);
            }
        }
    }
    /**-    Loop    -**/

    cv::imshow("img_1", images[0]);
    cv::imshow("img_2", images[1]);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
