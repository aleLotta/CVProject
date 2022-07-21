//segmentation 
//https://learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-rcnn-in-opencv-python-c/

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdlib.h>
#include <opencv2/dnn/dnn.hpp>


using namespace std;
using namespace cv;
using namespace cv::dnn;

vector<String> getOutputsNames(const Net& net);
void postprocess(Mat& frame, const vector<Mat>& outs);
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;        // Width of network's input image
int inpHeight = 416;       // Height of network's input image

// Classes names
String classes = {"Hands"};


int main(int argc, char** argv)
{
	// Give the configuration and the weight files 
    	String modelConfiguration = "yolov3_testing.cfg";
    	String modelWeights = "yolov3_training_last.weights";
    	
    	// Load Yolo Model
	//cv::dnn::Net net = cv::dnn::readNet("../../../Model/yolov3_training_last.weights", "../../../Model/yolov3_testing.cfg");
    	cv::dnn::Net net = cv::dnn::readNet(modelWeights, modelConfiguration, "Darknet");
    	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
    	
    	DetectionModel model = DetectionModel(net);
    	model.setInputParams(1/255.0, cv::Size(inpWidth, inpHeight), Scalar(), true);
    	
    	// gather images in a specified path
    	vector<String> images_path;
    	string folder = "/home/local/varoeli65246/Desktop/project/Dataset progetto CV - Hand detection _ segmentation/rgb/*.jpg";
    	glob(folder, images_path, false); 
    	
    	vector<int> classIds;
    	vector<Rect> boxes;
    	vector<float> scores;
    	
    	Mat img; 
    	for (int i = 0; i < images_path.size(); i++)
    	{
      		img = imread(images_path[i]);
      		model.detect(img, classIds, scores, boxes, 0.6, 0.4);
      		// Generate random colors
    		vector<Vec3b> colors;
    		for (size_t i = 0; i < 100; i++)
    		{
        		int b = theRNG().uniform(0, 256);
        		int g = theRNG().uniform(0, 256);
        		int r = theRNG().uniform(0, 256);
        		colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    		}
      		for (int j = 0; j<classIds.size(); j++)
    		{
    			rectangle(img, boxes[j], colors[j], 2);
    		}
    		
    		imshow("img", img);        
      		waitKey(0);
      	}
	
	return 0;
}

