QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++20
CONFIG += warn_off
QMAKE_CXXFLAGS += -Wall
QMAKE_CXXFLAGS += -Wno-comment

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    ../../modules/resnet/resnet_pt.cpp \
    ../../modules/yolov5/yolov5_onnx.cpp \
    main.cpp \
    mainwindow.cpp \

HEADERS += \
    ../../include/calculation.h \
    ../../include/local_feature.h \
    ../../modules/resnet/resnet_pt.h \
    ../../modules/yolov5/yolov5_onnx.h \
    common.h \
    mainwindow.h \

FORMS += \
    mainwindow.ui

# OpenCV
INCLUDEPATH += /usr/local/include/opencv4
LIBS += /usr/local/lib/libopencv_*.so

# libtorch
INCLUDEPATH += /home/rcxxx/WorkSpace/code/libtorch/include \
                /home/rcxxx/WorkSpace/code/libtorch/include/torch/csrc/api/include
LIBS += /home/rcxxx/WorkSpace/code/libtorch/lib/lib*.so \

## common
INCLUDEPATH += ../../include

## yolov5
INCLUDEPATH += ../../modules/yolov5

## resnet
INCLUDEPATH += ../../modules/resnet

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
