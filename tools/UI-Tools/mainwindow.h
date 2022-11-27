#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QImage>
#include <QLabel>
#include <QFileDialog>
#include <QString>
#include <QFileInfo>
#include <QDebug>

#include <opencv2/opencv.hpp>

#include "common.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_yolo_det_clicked();

    void on_pushButton_save_feature_clicked();

    void on_pushButton_load_file_clicked();

private:
    Ui::MainWindow *ui;

    cv::Mat frame;

    const std::string feature_data_path = "../../data/feature.csv";
    std::ofstream write_handle;

    // YOLO
    const std::string     yolo_model_path = "../../models/yolov5s-480.onnx";
    const std::string     classes_path = "../../models/classes.txt";
    const std::string     resnet50_model_path = "../../models/resnet50-rm-fc.pth";

    yolov5::Net     yolo;
    resnet::ResNet  resnet50;
    cv::Mat         crop_img;
};
#endif // MAINWINDOW_H
