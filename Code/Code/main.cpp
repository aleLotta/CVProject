/*** Libraries and Namespaces ***/

#include <opencv2/opencv.hpp>
#include <fstream>
#include <math.h>

using namespace cv;
using namespace std;
using namespace cv::dnn;


/*** Global Parameters ***/

const string CLASS_NAME = "hand";
// costants
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;
// text
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;
// colors
Scalar BLACK = Scalar(0, 0, 0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);
Scalar GREEN = Scalar(0, 255, 0);
Scalar PURPLE = Scalar(128, 0, 128);
Scalar ORANGE = Scalar(255, 117, 24);

vector<Scalar> colors = { BLUE, RED, GREEN, YELLOW, PURPLE, ORANGE };

/*** Pre-Processing ***/

vector<Mat> pre_process(const Mat& image, Net& net) {
    // Convert to blob
    Mat blob;
    blobFromImage(image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}


/*** Post-Processing ***/

Mat post_process(Mat& image, const vector<Mat>& outputs, const vector<Rect>& gtBoxes, vector<Rect>& predBoxes) {
    // Only 1 class id (0: hand)
    Point class_id;
    class_id.x = 0;
    class_id.y = 0;
    // Initialize vectors to hold respective outputs while unwrapping detections
    vector<float> confidences;
    vector<Rect> boxes;
    // Resizing factor
    float x_factor = image.cols / INPUT_WIDTH;
    float y_factor = image.rows / INPUT_HEIGHT;
    float* data = (float*)outputs[0].data;
    const int dimensions = 6;

    /* Filter Good Detections */

    // 25200 for default size 640
    const int rows = 25200;
    // Iterate through 25200 detections
    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        // Discard bad detections and continue
        if (confidence >= CONFIDENCE_THRESHOLD) {
            // Store class ID and confidence in the pre-defined respective vectors
            confidences.push_back(confidence);
            // Center
            float cx = data[0];
            float cy = data[1];
            // Box dimension
            float w = data[2];
            float h = data[3];
            // Bounding box coordinates
            int left = int((cx - 0.5 * w) * x_factor);
            int top = int((cy - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);
            // Store good detections in the boxes vector
            boxes.push_back(Rect(left, top, width, height));
        }
        // Jump to the next row
        data += dimensions;
    }

    /* Remove Overlapping Boxes */

    // Perform Non-Maximum Suppression and draw predictions
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    
    for (int i = 0; i < indices.size(); i++) {

        int idx = indices[i];
        Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Save detected Box
        predBoxes.push_back(Rect(left, top, width, height));
        // Draw bounding box
        rectangle(image, Point(left, top), Point(left + width, top + height), colors[i % colors.size()], 2 * THICKNESS);
    }

    return image;
}


/*** IoU Metric ***/

float bboxes_iou(Rect gtBox, Rect predBox) {
    
    // Coordinates for intersection rectangle
    int xA = max(gtBox.x, predBox.x);
    int yA = max(gtBox.y, predBox.y);
    int xB = min(gtBox.x + gtBox.width, predBox.x + predBox.width);
    int yB = min(gtBox.y + gtBox.height, predBox.y + predBox.height);
    
    // Area of intersection rectangle
    float interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1);
    
    // IoU as the intersection area of the two boxes divided by their union area
    float iou = interArea / float(gtBox.area() + predBox.area() - interArea);
    
    return iou;
}

float iou_calc(const Mat& input_image, const vector<Rect>& gtBoxes, const vector<Rect>& predBoxes) {

    int nMin = min(gtBoxes.size(), predBoxes.size());
    int nMax = max(gtBoxes.size(), predBoxes.size());
    int gtIndex = -1, predIndex = -1;
    float tempIou, bestIou, finalIou = 0;
    vector<int> predDone(predBoxes.size(), 0);
    vector<int> gtDone(gtBoxes.size(), 0);

    for (int k = 0; k < nMin; k++) {
        bestIou = 0;
        
        for (int i = 0; i < predBoxes.size(); i++) {
            if (predDone[i] != 1) {
                for (int j = 0; j < gtBoxes.size(); j++) {
                    if (gtDone[j] != 1) {
                        tempIou = bboxes_iou(predBoxes[i], gtBoxes[j]);
                        
                        if (tempIou > bestIou) {  
                            bestIou = tempIou;
                            predIndex = i;
                            gtIndex = j;
                        }
                    }
                }
            }
        }

        predDone[predIndex] = 1;
        gtDone[gtIndex] = 1;
        finalIou += bestIou;
    }

    return finalIou / nMax;
}


/*** Hand Segmentation ***/

Mat hand_segmentation(Mat& frame, vector<Rect> boxes, Mat& mask) {
    
    Mat final_img; frame.copyTo(final_img);
    Mat blur;
    //bilateralFilter(frame, blur, 9, 100, 100);
    //GaussianBlur(frame, blur, Size(9, 9), 0, 0);

    for (int t = 0; t < boxes.size(); t++) {
        Rect box = boxes[t];
        int x = box.x;
        int y = box.y;
        int w = box.width;
        int h = box.height;

        Rect coordinates = Rect(x, y, w, h);

        Mat result; // segmentation result (4 possible values)
        Mat bgModel, fgModel; // the models (internally used)    

        // GrabCut segmentation
        grabCut(frame,    // input image
            result,   // segmentation result
            coordinates,// rectangle containing foreground
            bgModel, fgModel, // models
            1,        // number of iterations
            cv::GC_INIT_WITH_RECT);

        cv::Mat foreground(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        
        // Draw background (black) or hand (white) pixels on foreground image
        for (int i = 0; i < frame.rows; i++)
            for (int j = 0; j < frame.cols; j++)
                if (result.at<uchar>(i, j) == 0 || result.at<uchar>(i, j) == 2) {
                    foreground.at<Vec3b>(i,j) = Vec3b(0,0,0);
                    result.at<uchar>(i,j) = 0;
                }
                else {   
                    foreground.at<Vec3b>(i,j)[0] = colors[t][0];
                    foreground.at<Vec3b>(i,j)[1] = colors[t][1];
                    foreground.at<Vec3b>(i,j)[2] = colors[t][2];
                    result.at<uchar>(i,j) = 255;
                }
                
        // Computation for the final mask
        mask = mask + result;

        // Generate output image
        cv::Mat foreground(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0));

        // Apply a color to the mask
        Mat temp(frame.rows, frame.cols, CV_8UC3, colors[t]);
        temp.copyTo(foreground, result); // bg pixels not copied

        //imshow("Foreground.jpg", foreground);
        //waitKey();

        //Mat final_img;
        // Merge the original image with the mask
        addWeighted(final_img, 1, foreground, 0.5, 0.0, final_img);
        imshow("Overlap", final_img);
        waitKey();

    }

    return final_img;
}


/*** Pixel Accuracy Metric ***/

float pixel_accuracy(Mat gT, Mat det_img) {

    float true_neg = 0, true_pos = 0;
    for (int i = 0; i < gT.cols; i++) {
        for (int j = 0; j < gT.rows; j++) {
            if (gT.at<uchar>(j, i) == 0 && det_img.at<uchar>(j, i) == 0) {
                true_neg++;
            }
            if (gT.at<uchar>(j, i) == 255 && det_img.at<uchar>(j, i) == 255) {
                true_pos++;
            }
        }
    }

    float total = gT.total();

    float pA = (true_neg + true_pos) / total;
    return pA;
}


/*** -> Main Function <- ***/

int main() {
    // Load images, labels and masks
    Mat frame;
    vector<string> image_paths;
    vector<string> label_paths;
    vector<string> mask_paths;
    glob("Dataset progetto CV - Hand detection _ segmentation/rgb/*.jpg", image_paths, false); // 30 imaes
    glob("Dataset progetto CV - Hand detection _ segmentation/det/*.txt", label_paths, false); // 30 labels
    glob("Dataset progetto CV - Hand detection _ segmentation/mask/*.png", mask_paths, false); // 30 masks

    // Load model
    Net net = readNet("Model5/last425m.onnx");

    // Process images and labels
    vector<Mat> detections;
    ifstream file;
    vector<Rect> labels;
    vector<Rect> boxes;
    Mat frame_copy;
    Mat gT_mask;
    int x, y, w, h;
    
    for (int i = 0; i < image_paths.size(); i++) {

        labels.clear();
        boxes.clear();

        frame = imread(image_paths[i]);
        file = ifstream(label_paths[i]);
        while (file >> x >> y >> w >> h)
            labels.push_back(Rect(x, y, w, h));

        detections = pre_process(frame, net);

        frame.copyTo(frame_copy); // deep copy of image

        Mat img = post_process(frame_copy, detections, labels, boxes);
        //imshow("Output", img);
        //waitKey(0);
        String name = "Detection2/" + std::to_string(i + 1) + ".jpg";
        imwrite(name, img);
        
        float iou = iou_calc(frame, labels, boxes);

        float iou = iou_calc(frame, labels, boxes);


        // Segmentation
        gT_mask = imread(mask_paths[i], IMREAD_GRAYSCALE);
        Mat mask_img = Mat::zeros(frame.rows, frame.cols, CV_8U);
        Mat final_img = hand_segmentation(frame, boxes, mask_img);

        std::string savingName = "Segmentation2/" + std::to_string(i + 1) + ".jpg";
        imwrite(savingName, final_img);

        // Caluclate pixel accuracy
        float frame_PA = pixel_accuracy(gT_mask, mask_img);

        fstream pA_results;

        pA_results.open("metrics_results.txt", fstream::app);
        if (pA_results.is_open()) {
            pA_results << "iou" + to_string(i) + "\n" + to_string(iou) + "\n\n";
            pA_results << "pa_mask" + to_string(i) + "\n" + to_string(frame_PA) + "\n\n";
            pA_results.close();
            
        }
    }
    
    return 0;
}