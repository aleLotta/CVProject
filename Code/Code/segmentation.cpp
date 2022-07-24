#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat src_gray;
int thresh = 100;
RNG rng(12345);

void thresh_callback(int, void* );

Mat K_Means(Mat Input, int K) {
	Mat samples(Input.rows * Input.cols, Input.channels(), CV_32F);
	for (int y = 0; y < Input.rows; y++)
		for (int x = 0; x < Input.cols; x++)
			for (int z = 0; z < Input.channels(); z++)
				if (Input.channels() == 3) {
					samples.at<float>(y + x * Input.rows, z) = Input.at<Vec3b>(y, x)[z];
				}
				else {
					samples.at<float>(y + x * Input.rows, z) = Input.at<uchar>(y, x);
				}

	Mat labels;
	int attempts = 10;
	Mat centers;
  TermCriteria criteria = TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0);
	kmeans(samples, K, labels, criteria, attempts, KMEANS_PP_CENTERS, centers);


	Mat new_image(Input.size(), Input.type());
	for (int y = 0; y < Input.rows; y++)
		for (int x = 0; x < Input.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * Input.rows, 0);
			if (Input.channels()==3) {
				for (int i = 0; i < Input.channels(); i++) {
					new_image.at<Vec3b>(y, x)[i] = centers.at<float>(cluster_idx, i);
				}
			}
			else {
				new_image.at<uchar>(y, x) = centers.at<float>(cluster_idx, 0);
			}
		}
	//imshow("clustered image", new_image);
	return new_image;
}

int main(int argc, char** argv)
{
  int x = 631;
  int y = 318;
  int w = 217;
  int h = 122;
 
  /*int x = 89;
  int y = 90;
  int w = 68;
  int h = 69;*/
  
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
 
  /*Mat blur;
  bilateralFilter(rectangle, blur, 9, 150, 150);
  imshow("Prova1", blur);
  waitKey(0);*/
 
 
 
  
  Mat gray_rect;
  cvtColor( rectangle, gray_rect, COLOR_BGR2GRAY );

  gray_rect.copyTo(src_gray);
  
  
  // K-MEANS
  //GaussianBlur( src_gray, src_gray, Size(9,9), 0);
  Mat clusteredImg = K_Means(src_gray, 3);
  imshow("clusters", clusteredImg);
  waitKey(0);
  
  uchar central_pixel = clusteredImg.at<uchar>(clusteredImg.rows/2,clusteredImg.cols/2);
  
  Mat otsu_img;
  threshold(clusteredImg, otsu_img, 0, 255, THRESH_OTSU);
  imshow("Otsu", otsu_img);
  waitKey();
  
  
  
  
  // OTSU THRESHOLD
  //Mat thresh;
  /*Mat temp_img, dst_img, t_img;
  
  dst_img = Mat::zeros(gray_rect.size(), gray_rect.type());
  t_img = Mat::zeros(gray_rect.size(), gray_rect.type());
  
  int rect_h = floor(rectangle.rows/2);
  int rect_w = floor(rectangle.cols/3);

  for (int k = 0;k < 3; k++){
    for (int g = 0; g < 2; g++) {
      temp_img = Mat::zeros(gray_rect.size(), gray_rect.type());
      for (int i = g*rect_w; i < rect_w*(g+1); i++){
        for (int j = k*rect_h; j < rect_h*(k+1); j++){
          temp_img.at<uchar>(i,j) = 255;
        }
      }
      
      gray_rect.copyTo(temp_img, temp_img);
      
      //GaussianBlur(temp_img,t_img,Size(9,9),0);
      bilateralFilter(temp_img, t_img, 5, 75, 75);
      threshold(t_img, t_img, 0, 255, THRESH_OTSU);
      
      dst_img = dst_img + t_img;
      
      imshow("k",dst_img);
      waitKey();
    }
  } 
  imshow("Otsu", dst_img);
  waitKey();*/
 
  
  // CANNY + CONTOURS
  /*const int max_thresh = 255;
  const char* source_window = "Source";
  namedWindow(source_window);
  createTrackbar( "Canny thresh:", source_window, &thresh, max_thresh, thresh_callback );
  thresh_callback(0,0);
  waitKey();*/

  

  return 0;
}

void thresh_callback(int, void* )
{
    Mat canny_output;
    //Canny( src_gray, canny_output, thresh, thresh*2 );
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    //findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );
    findContours( src_gray, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE );
    //Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    threshold(src_gray, canny_output, 200, 255, THRESH_BINARY);
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( drawing, contours, (int)i, color, FILLED, 8, hierarchy);
        //drawContours( drawing, contours, (int)i, color, 2, LINE8, hierarchy);
    }
    imshow( "Contours", drawing );
}