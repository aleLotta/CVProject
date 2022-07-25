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
    imshow("Source", img);
    waitKey();

    Mat box = Mat::zeros(h, w, img.type());

    int k = 0;
    int z = 0;
    for (int i = x; i < x + w; i++) {
        z = 0;
        for (int j = y; j < y + h; j++) {
            box.at<Vec3b>(z, k) = img.at<Vec3b>(j, i);
            z++;
        }
        k++;
    }
    imshow("Bounding Box", box);
    waitKey();

    // ALTERNATIVE SOLUTION
    Rect coordinates = Rect(x, y, w, h);

    Mat result; // segmentation result (4 possible values)
    Mat bgModel, fgModel; // the models (internally used)

    // GrabCut segmentation
    grabCut(img,    // input image
        result,   // segmentation result
        coordinates,// rectangle containing foreground
        bgModel, fgModel, // models
        1,        // number of iterations
        cv::GC_INIT_WITH_RECT);

    // Get the pixels marked as likely foreground
    cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
    // Generate output image
    cv::Mat foreground(img.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    //cv::Mat background(image.size(),CV_8UC3,cv::Scalar(255,255,255));
    Mat temp(img.rows, img.cols, CV_8UC3, Scalar(255, 0, 0));
    temp.copyTo(foreground, result); // bg pixels not copied

    /*img.copyTo(foreground, result);
    Mat box = Mat::zeros(h, w, img.type());

    int k = 0;
    int z = 0;
    for (int i = x; i < x + w; i++) {
        z = 0;
        for (int j = y; j < y + h; j++) {
            box.at<Vec3b>(z, k) = foreground.at<Vec3b>(j, i);
            z++;
        }
        k++;
    }
    imshow("Bounding Box", box);
    waitKey();

    Mat blur;
    //bilateralFilter(box, blur, 9, 100, 150);
    GaussianBlur( box, blur, Size(9,9), 0);
    Mat clusteredImg = K_Means(blur, 3);
    imshow("clusters", clusteredImg);
    waitKey(0);*/

    imshow("Foreground.jpg", foreground);
    waitKey();

    /* // draw rectangle on original image
    cv::rectangle(img, coordinates, cv::Scalar(255, 255, 255), 1);
    Mat background;
    img.copyTo(background, ~result);
    imshow("Background.jpg", background);
    waitKey(); */

    Mat final_img;
    addWeighted(img, 1, foreground, 0.5, 0.0, final_img);
    imshow("Overlap", final_img);
    waitKey();

    return 0;
}