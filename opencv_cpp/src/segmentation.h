#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace cv::dnn;
using namespace std;


// Pre-Processing
vector<Mat> PreProcess(const Mat&, Net&);

// Post-Processing
vector<Rect> PostProcess(Mat&, const vector<Mat>&);

// IoU Metric
float StandardIouMetric(const Rect&, const Rect&);
float ImageIouMetric(const vector<Rect>&, const vector<Rect>&);

// Hand Segmentation
Mat HandSegmentation(const Mat&, Mat&, const vector<Rect>&);

// Pixel Accuracy Metric
float PixelAccuracy(const Mat&, const Mat&);


#endif