/*** 1) Libraries and Namespaces ***/

#include <opencv2/opencv.hpp>
#include <fstream>
using namespace cv;
using namespace std;
using namespace cv::dnn;


/*** 2) Global Parameters ***/

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

vector<Scalar> colours = { BLUE, RED, GREEN, YELLOW };


/*** 3) Draw Label ***/

void draw_label(Mat& input_image, string label, int left, int top)
{
    // Display the label at the top of the bounding box
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    // Top left corner
    Point tlc = Point(left, top);
    // Bottom right corner
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw white rectangle
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}


/*** 4) Pre-Processing ***/

vector<Mat> pre_process(Mat& input_image, Net& net)
{
    // Convert to blob
    Mat blob;
    blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}


/*** 5) Post-Processing ***/

Mat post_process(Mat& input_image, vector<Mat>& outputs, vector<Rect> gtBoxes, vector<Rect>& detBoxes)
{
    // Only 1 class id (0: hand)
    Point class_id;
    class_id.x = 0;
    class_id.y = 0;
    // Initialize vectors to hold respective outputs while unwrapping detections
    vector<float> confidences;
    vector<Rect> boxes;
    // Resizing factor
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float* data = (float*)outputs[0].data;
    const int dimensions = 6;

    /* A. Filter Good Detections */

    // 25200 for default size 640
    const int rows = 25200;
    // Iterate through 25200 detections
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
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

    /* B. Remove Overlapping Boxes */

    // Perform Non-Maximum Suppression and draw predictions
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    // Draw ground truth bounding box
    for (int i = 0; i < gtBoxes.size(); i++)
        rectangle(input_image, Point(gtBoxes[i].x, gtBoxes[i].y), Point(gtBoxes[i].x + gtBoxes[i].width, gtBoxes[i].y + gtBoxes[i].height), RED, 3 * THICKNESS);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Save detected Box
        detBoxes.push_back(Rect(left, top, width, height));
        // Draw bounding box
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3 * THICKNESS);
        // Get the label for the class name and its confidence
        string label = format("%.2f", confidences[idx]);
        // label = CLASS_NAME + ":" + label;
        // Draw class labels
        draw_label(input_image, label, left, top);
    }
    return input_image;
}


/*** 6) IoU Metric ***/

float bboxes_iou(Rect gtBox, Rect predBox)
{
    // Coordinates for intersection rectangle
    int xA = max(gtBox.x, predBox.x);
    int yA = max(gtBox.y, predBox.y);
    int xB = min(gtBox.x + gtBox.width, predBox.x + predBox.width);
    int yB = min(gtBox.y + gtBox.height, predBox.y + predBox.height);
    // Area of intersection rectangle
    float interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1);
    // Area of predicted and ground truth bounding boxes
    //float gtBoxArea = (gtBox.width + 1) * (gtBox.height + 1);
    //float predBoxArea = (predBox.width + 1) * (predBox.height + 1);
    // IoU as the intersection area of the two boxes divided by their union area
    float iou = interArea / float(gtBox.area() + predBox.area() - interArea);
    return iou;
}


/*** 7) Hand Segmentation ***/

void hand_segmentation(Mat& frame, vector<Rect> boxes) {
    /* SEGMENTATION */
    Mat final_img; frame.copyTo(final_img);

    for (int t = 0; t < boxes.size(); t++) {
        Rect box = boxes[t];
        int x = box.x;
        int y = box.y;
        int w = box.width;
        int h = box.height;

        // ALTERNATIVE SOLUTION
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

        // Get the pixels marked as likely foreground
        cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
        // Generate output image
        cv::Mat foreground(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        //cv::Mat background(image.size(),CV_8UC3,cv::Scalar(255,255,255));
        Mat temp(frame.rows, frame.cols, CV_8UC3, colours[t]);
        temp.copyTo(foreground, result); // bg pixels not copied

        imshow("Foreground.jpg", foreground);
        //waitKey();

        /* // draw rectangle on original image
        cv::rectangle(img, coordinates, cv::Scalar(255, 255, 255), 1);
        Mat background;
        img.copyTo(background, ~result);
        imshow("Background.jpg", background);
        waitKey(); */

        //Mat final_img;
        addWeighted(final_img, 1, foreground, 0.5, 0.0, final_img);
        imshow("Overlap", final_img);
        waitKey();

    }
}

/*** -> Main Function ***/

int main()
{
    // Load images and labels
    Mat frame;
    vector<string> image_paths;
    vector<string> label_paths;
    glob("Dataset progetto CV - Hand detection _ segmentation/rgb/*.jpg", image_paths, false); // 30 imaes
    glob("Dataset progetto CV - Hand detection _ segmentation/det/*.txt", label_paths, false); // 30 labels
    // string path = "val/images/CARDS_LIVINGROOM_B_T_frame_0504_jpg.rf.50fe772fe60ff8aec573157df5824a5a.jpg";
    // string path = "val/images/2.jpg";
    // frame = imread(path);
    // Load model
    Net net;
    net = readNet("best.onnx");
    // Process images and labels
    vector<Mat> detections;
    ifstream file;
    vector<Rect> labels;
    vector<Rect> boxes;

    int x, y, w, h;
    for (int i = 0; i < image_paths.size(); i++)
    {
        labels.clear();
        frame = imread(image_paths[i]);
        file = ifstream(label_paths[i]);
        while (file >> x >> y >> w >> h) // mickel <3
            labels.push_back(Rect(x, y, w, h));
        detections = pre_process(frame, net);
        Mat frame_copy; frame.copyTo(frame_copy); // deep copy of image
        // Mat img = post_process(frame_copy, detections);
        boxes.clear();
        Mat img = post_process(frame_copy, detections, labels, boxes);
        imshow("Output", img);
        waitKey(0);

        hand_segmentation(frame, boxes);

    }
    return 0;
}