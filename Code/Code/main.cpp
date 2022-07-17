#include <opencv2/opencv.hpp>

int main(int, char **)
{
    Net net = cv::dnn::readNet("yolov3_training_last.weights", "yolov3_testing.cfg");
    
    return 0;
}