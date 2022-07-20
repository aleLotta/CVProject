#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{ 
  int x = 631;
  int y = 318;
  int w = 217;
  int h = 122;
  
  Mat img = imread("01.jpg");
  cout<<img.size()<<endl;
  
  Mat box = Mat::zeros(img.rows, img.cols, img.type());
  cout<<box.size()<<endl;
  
  for (int k = x; k < w+x; k++){
    for (int z = y; z < h+y; z++){
        box.at<Vec3b>(z,k) = img.at<Vec3b>(z,k);
    }
  }
  
  imshow("Prova", box);
  //waitKey(0);
  
  
  //bilateralFilter
  //Mat blur = Mat::zeros(img.rows, img.cols, img.type());
  Mat gray_box;
  cvtColor( box, gray_box, COLOR_BGR2GRAY );
  /*Mat blur;
  bilateralFilter(gray_box, blur, 5, 200, 200);
  imshow("Prova1", blur);
  waitKey(0);*/
  
  //rectangle 
  Mat rectangle = Mat::zeros(h, w, gray_box.type());
  int i = 0;
  int j = 0;
  for (int k = x; k < w+x; k++){
    j = 0;
    for (int z = y; z < h+y; z++){
        rectangle.at<uchar>(j,i) = gray_box.at<uchar>(z,k);
        j++;
    }
    i++;
  }
  
  imshow("rect", rectangle);
  waitKey(0);
  
  //Otsu
  Mat thresh;
  long double thres = cv::threshold(rectangle, thresh, 0,255, THRESH_OTSU);
  
  imshow("Prova2", thresh);
  waitKey(0);
  //Canny edge
  //
  
  return 0;
}