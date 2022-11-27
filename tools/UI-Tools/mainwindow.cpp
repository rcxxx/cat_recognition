#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , write_handle(feature_data_path, std::ios_base::app)
    , yolo(yolo_model_path, classes_path)
    , resnet50(resnet50_model_path)
{
    ui->setupUi(this);

//    ui->pushButton_save_feature->setEnabled(false);
//    ui->pushButton_save_feature->setEnabled(false);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_yolo_det_clicked()
{
    if (!frame.empty()) {
        cv::Mat src_img = this->frame;
        std::vector<yolov5::Detection> result =  yolo.detect(src_img);

        cv::Mat crop_tmp;
        int cat_cnt = 0;
        for(const auto &idx : result){
            if(idx.class_id == 15 || idx.class_id == 16){
                auto bbox = idx.bbox;
                cv::rectangle(src_img, bbox, cv::Scalar(0, 255, 255), 2);
                cv::rectangle(src_img, cv::Point(bbox.x, bbox.y + 10), cv::Point(bbox.x + bbox.width, bbox.y), cv::Scalar(0, 255, 255), cv::FILLED);
                cv::putText(src_img, yolo.classList()[15], cv::Point(bbox.x, bbox.y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

                crop_tmp = rmRoiBoundary(src_img, bbox, true, 0.25);
                cat_cnt += 1;
            }
        }
        if (cat_cnt == 1){
            ui->pushButton_save_feature->setEnabled(true);
            this->crop_img = crop_tmp;
            cv::Mat rgb_img;
            cv::cvtColor(crop_tmp, rgb_img, cv::COLOR_BGR2RGB);
            cv::resize(rgb_img,rgb_img, cv::Size(224,224));
            QImage q_frame = QImage((const uchar*)rgb_img.data, rgb_img.cols, rgb_img.rows, rgb_img.cols*rgb_img.channels(), QImage::Format_RGB888);
            ui->label_crop->setPixmap(QPixmap::fromImage(q_frame));

//            ui->pushButton_save_feature->setEnabled(true);
        }

//        ui->pushButton_yolo_det->setEnabled(false);
    }
}

void MainWindow::on_pushButton_save_feature_clicked()
{
    write_handle = std::ofstream(feature_data_path, std::ios_base::app);
    QString name = ui->lineEdit_feature_name->text();
    if (!name.isEmpty()){
        ui->label_save_file->setText(name);

        auto crop_feature = resnet50.inference(this->crop_img);
        auto ptr_f = crop_feature.data_ptr<float>();
        if (crop_feature.sizes()[0]){
            this->write_handle << name.toStdString() << " ";
            for (size_t i = 0; i < crop_feature.sizes()[1]; ++i){
                this->write_handle << *ptr_f++ << " ";
            }
            this->write_handle << "\n";
        }
    }
    this->write_handle.close();
}

void MainWindow::on_pushButton_load_file_clicked()
{
    QString file = QFileDialog::getOpenFileName(this,tr("open iamge"),"../UI-Tools/");
    if (!file.isEmpty()){
        QFileInfo f_info = QFileInfo(file);
        QString f_name = f_info.fileName();
        QString f_suffix = f_info.suffix();
        if (f_suffix == "jpg" || f_suffix == "png" || f_suffix == "jpeg"){
            cv::Mat src_img = cv::imread(file.toLatin1().data());
            cv::resize(src_img, src_img, cv::Size(720, 480));
            src_img.copyTo(this->frame);

            cv::Mat rgb_img;
            cv::cvtColor(frame, rgb_img, cv::COLOR_BGR2RGB);
            QImage q_frame = QImage((const uchar*)rgb_img.data, rgb_img.cols, rgb_img.rows, rgb_img.cols*rgb_img.channels(), QImage::Format_RGB888);
            ui->label_src->setPixmap(QPixmap::fromImage(q_frame));
            ui->label_load_file->setText("loaded:" + f_name);

//            ui->pushButton_yolo_det->setEnabled(true);
        }
    }
}
