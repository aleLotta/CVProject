#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdlib.h>

//void Erosion( int, void* );


using namespace cv;
using namespace std;

RNG rng(12345);


int main(int argc, char** argv)
{ 
  	int x = 631;
  	int y = 318;
  	int w = 217;
  	int h = 122;
  
  	Mat img = imread("01.jpg");
  	//cout<<img.size()<<endl;
  
  	Mat box = Mat::zeros(img.rows, img.cols, img.type());
  	//cout<<box.size()<<endl;
  
  	for (int k = x; k < w+x; k++){
    		for (int z = y; z < h+y; z++){
        	box.at<Vec3b>(z,k) = img.at<Vec3b>(z,k);
    		}
  	}
  
  	//imshow("Prova", box);
  	//waitKey(0);
 
  
  	//rectangle 
  	Mat rectangle = Mat::zeros(h, w, box.type());
  	int i = 0;
  	int j = 0;
  	for (int k = x; k < w+x; k++){
    		j = 0;
    		for (int z = y; z < h+y; z++){
        		rectangle.at<Vec3b>(j,i) = box.at<Vec3b>(z,k);
        		j++;
    		}
    		i++;
  	}
  
  	imshow("rect", rectangle);
 	waitKey(0);
 	
 	Mat kernel = (Mat_<float>(3,3) <<
                  1,  1, 1,
                  1, -8, 1,
                  1,  1, 1); // an approximation of second derivative, a quite strong kernel
    	
    	Mat imgLaplacian;
    	filter2D(rectangle, imgLaplacian, CV_32F, kernel);
    	Mat sharp;
    	rectangle.convertTo(sharp, CV_32F);
    	Mat imgResult = sharp - imgLaplacian;
    	// convert back to 8bits gray scale
    	imgResult.convertTo(imgResult, CV_8UC3);
    	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    	// imshow( "Laplace Filtered Image", imgLaplacian );
    	imshow( "New Sharped Image", imgResult );
  	waitKey(0);

	Mat img_canny;
  	Canny(imgResult, img_canny, 2000, 3000, 5);
  	imshow( "New Sharped Image", img_canny);
  	waitKey(0);
  	
  	Mat erosion_dst;
  	Mat dilation_dst;
  	dilate( img_canny, dilation_dst, Mat() );
  	erode( dilation_dst, erosion_dst, Mat() );
 	imshow( "erosion", erosion_dst );
 	//imshow( "dilation", dilation_dst );
 	waitKey(0);
  	return 0;
  	
  	return 0;
}







	


/*
//soluzione 1 : canny + erosion
Mat img_canny;
  	Canny(bw, img_canny, 1000, 2000, 5);
  	imshow( "New Sharped Image", img_canny);
  	waitKey(0);
  	
  	vector<vector<Point> > contours;
    	vector<Vec4i> hierarchy;
    	findContours( img_canny, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );
    	Mat drawing = Mat::zeros( img_canny.size(), CV_8UC3 );
    	for( size_t i = 0; i< contours.size(); i++ )
    	{
        	Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        	drawContours( drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0 );
    	}
    	imshow( "Contours", drawing );
    	waitKey(0);
  
  	Mat erosion_dst;
  	Mat dilation_dst;
  	erode( drawing, erosion_dst, Mat() );
  	dilate( drawing, dilation_dst, Mat() );
 	imshow( "erosion", erosion_dst );
 	imshow( "dilation", dilation_dst );
 	waitKey(0);
  	return 0;
void Erosion( int, void* )
{
  int erosion_type = 0;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
  Mat element = getStructuringElement( erosion_type,
                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                       Point( erosion_size, erosion_size ) );
  erode( src, erosion_dst, element );
  imshow( "Erosion Demo", erosion_dst );
}

// soluzione 2 : watershed
// Create a kernel that we will use to sharpen our image
    	Mat kernel = (Mat_<float>(3,3) <<
                  1,  1, 1,
                  1, -8, 1,
                  1,  1, 1); // an approximation of second derivative, a quite strong kernel
    	
    	Mat imgLaplacian;
    	filter2D(rectangle, imgLaplacian, CV_32F, kernel);
    	Mat sharp;
    	rectangle.convertTo(sharp, CV_32F);
    	Mat imgResult = sharp - imgLaplacian;
    	// convert back to 8bits gray scale
    	imgResult.convertTo(imgResult, CV_8UC3);
    	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    	// imshow( "Laplace Filtered Image", imgLaplacian );
    	imshow( "New Sharped Image", imgResult );
  	waitKey(0);
  	
  	Mat bw;
    	cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    	threshold(bw, bw, 200, 255, THRESH_BINARY | THRESH_OTSU);
    	imshow("Binary Image", bw);
    	waitKey(0);
    	
    	
    	// Perform the distance transform algorithm
    	Mat dist;
    	distanceTransform(bw, dist, DIST_L2, 5);
    	// Normalize the distance image for range = {0.0, 1.0}
    	// so we can visualize and threshold it
    	normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    	imshow("Distance Transform Image", dist);
	waitKey(0);
	
	// Threshold to obtain the peaks
    	// This will be the markers for the foreground objects
    	threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
    	// Dilate a bit the dist image
    	Mat kernel1 = Mat::ones(3, 3, CV_8U);
    	dilate(dist, dist, kernel1);
    	imshow("Peaks", dist);
    	waitKey(0);
    	
    	// Create the CV_8U version of the distance image
    	// It is needed for findContours()
    	Mat dist_8u;
    	dist.convertTo(dist_8u, CV_8U);
    	// Find total markers
    	vector<vector<Point> > contours;
    	findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    	// Create the marker image for the watershed algorithm
    	Mat markers = Mat::zeros(dist.size(), CV_32S);
    	// Draw the foreground markers
    	for (size_t i = 0; i < contours.size(); i++)
    	{
        	drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i)+1), -1);
    	}
    	// Draw the background marker
    	circle(markers, Point(5,5), 3, Scalar(255), -1);
    	Mat markers8u;
    	markers.convertTo(markers8u, CV_8U, 10);
    	imshow("Markers", markers8u);
    	waitKey(0);
    	
    	// Perform the watershed algorithm
    	watershed(imgResult, markers);
    	Mat mark;
    	markers.convertTo(mark, CV_8U);
    	bitwise_not(mark, mark);
    	//    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
    	// image looks like at that point
    	// Generate random colors
    	vector<Vec3b> colors;
    	for (size_t i = 0; i < contours.size(); i++)
    	{
        	int b = theRNG().uniform(0, 256);
        	int g = theRNG().uniform(0, 256);
        	int r = theRNG().uniform(0, 256);
        	colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    	}
    	// Create the result image
    	Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    	// Fill labeled objects with random colors
   	for (int i = 0; i < markers.rows; i++)
    	{
        	for (int j = 0; j < markers.cols; j++)
        	{
            		int index = markers.at<int>(i,j);
            		if (index > 0 && index <= static_cast<int>(contours.size()))
            		{
                	dst.at<Vec3b>(i,j) = colors[index-1];
            		}
        	}
    	}
    	// Visualize the final image
    	imshow("Final Result", dst);
    	waitKey(0);
*/
