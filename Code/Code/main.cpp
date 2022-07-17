#include <opencv2/opencv.hpp>

int main(int, char **)
{
    auto net = cv::dnn::readNet("yolov5s.onnx");
    
    return 0;
}