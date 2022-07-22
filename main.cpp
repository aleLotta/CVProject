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
    	string folder = "/home/local/varoeli65246/Desktop/project/Dataset progetto CV - Hand detection _ segmentation/rgb/01.jpg";
    	glob(folder, images_path, false); 
    	
    	vector<int> classIds;
    	vector<Rect> boxes;
    	vector<float> scores;
    	
    	Mat img;
    	int img_width = img.cols;
      	int img_height = img.rows;
    	Mat frame = Mat::zeros(img_height, img_width, img.type());
    	for (int i = 0; i < images_path.size(); i++)
    	{
      		img = imread(images_path[i]);
      		img.copyTo(frame);
      		model.detect(frame, classIds, scores, boxes, 0.6, 0.4);
      		// Generate random colors
    		vector<Vec3b> colors;
    		for (size_t i = 0; i < 100; i++)
    		{
        		int b = theRNG().uniform(0, 256);
        		int g = theRNG().uniform(0, 256);
        		int r = theRNG().uniform(0, 256);
        		colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    		}
    		
    		Mat rec;
      		for (int j = 0; j<classIds.size(); j++)
    		{
    			rectangle(frame, boxes[j], colors[j], 2);
    			//cout<<boxes[j].x<<endl;
    			//cout<<boxes[j].y<<endl;
    			//cout<<boxes[j].width<<endl;
    			//cout<<boxes[j].height<<endl;
    			rec = Mat::zeros(boxes[j].height, boxes[j].width, frame.type());
    			int r = 0;
  			int t = 0;
  			for (int k = boxes[j].x; k < boxes[j].width+boxes[j].x; k++)
  			{
    				t = 0;
    				for (int z = boxes[j].y; z < boxes[j].height+boxes[j].y; z++)
    				{
        				rec.at<uchar>(t,r) = frame.at<uchar>(z,k);
        				t++;
    				}
    				r++;
  			}
    		}
    		
    		imshow("frame", frame);        
      		waitKey(0);
      		imshow("rec", rec);        
      		waitKey(0);
      		
  		
      		
      	}
	/*
	// Create a kernel that we will use to sharpen our image
    	Mat kernel = (Mat_<float>(3,3) <<
                  	1,  1, 1,
                  	1, -8, 1,
                  	1,  1, 1); // an approximation of second derivative, a quite strong kernel
    	// do the laplacian filtering as it is
    	// well, we need to convert everything in something more deeper then CV_8U
    	// because the kernel has some negative values,
    	// and we can expect in general to have a Laplacian image with negative values
    	// BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    	// so the possible negative number will be truncated
    	Mat imgLaplacian;
    	filter2D(img, imgLaplacian, CV_32F, kernel);
    	Mat sharp;
    	img.convertTo(sharp, CV_32F);
    	Mat imgResult = sharp - imgLaplacian;
    	// convert back to 8bits gray scale
    	imgResult.convertTo(imgResult, CV_8UC3);
    	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    	// imshow( "Laplace Filtered Image", imgLaplacian );
    	imshow( "New Sharped Image", imgResult );
	waitKey(0);
	
	Mat bw;
    	cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    	threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
    	imshow("Binary Image", bw);
    	waitKey(0);
    	
    	// Perform the distance transform algorithm
    	Mat dist;
    	distanceTransform(bw, dist, DIST_L2, 3);
    	// Normalize the distance image for range = {0.0, 1.0}
    	// so we can visualize and threshold it
    	normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    	imshow("Distance Transform Image", dist);
	*/
	return 0;
}

void postprocess(Mat& frame, const vector<Mat>& outs)
