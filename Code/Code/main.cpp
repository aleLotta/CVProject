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
Scalar BLACK = Scalar(0,0,0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0,0,255);


float boundingBoxes_IoU(vector<Rect> predBox, vector<Rect> gTBox);

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

vector<Mat> pre_process(Mat &input_image, Net &net)
{
    // Convert to blob
    Mat blob;
    blobFromImage(input_image, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}


/*** 5) Post-Processing ***/

Mat post_process(Mat &input_image, vector<Mat> &outputs, vector<Rect> &boxes)
{
    // Only 1 class id (0: hand)
    Point class_id;
    class_id.x = 0;
    class_id.y = 0;
    // Initialize vectors to hold respective outputs while unwrapping detections
    vector<float> confidences;
    //vector<Rect> boxes;
    // Resizing factor
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float *data = (float *)outputs[0].data;
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
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3*THICKNESS);
        // Get the label for the class name and its confidence
        string label = format("%.2f", confidences[idx]);
        // label = CLASS_NAME + ":" + label;
        // Draw class labels
        draw_label(input_image, label, left, top);
    }
    return input_image;
}


/*** 6) Main Function ***/

int main()
{
    // Load image
    Mat frame;
    vector<cv::String> fn;
    glob("../../Dataset progetto CV - Hand detection _ segmentation/rgb/*.jpg", fn, false);

    vector<Mat> images;
    
    // Load model
    Net net;
    net = readNet("../../best.onnx");
    // Process the image
    vector<Mat> detections;
    
    size_t count = fn.size(); //number of png files in images folder
    for (size_t i=0; i<count; i++){
        frame = imread(fn[i]);
        detections = pre_process(frame, net);
        Mat frame_copy; frame.copyTo(frame_copy); // deep copy
        vector<Rect> boxes;
        Mat img = post_process(frame_copy, detections, boxes);
        imshow("Output", img);
        waitKey(0);
    }
    
    //string path = "../../Dataset progetto CV - Hand detection _ segmentation/rgb/02.jpg";
    // string path = "val/images/2.jpg";
    //frame = imread(path);
    
    
    // read groundTruth
    vector<cv::String> fn;
    glob("../../Dataset progetto CV - Hand detection _ segmentation/det/*.txt", fn, false);
    for (size_t i = 0; i < fn.size(); i++){
      newfile.open(fn[i],ios::in); //open a file to perform read operation using file object
      if (newfile.is_open()){ //checking whether the file is open
        string tp;
        while(getline(newfile, tp)){ //read data from file object and put it into string.
          cout << tp << "\n"; //print the data of the string
        }
        newfile.close(); //close the file object.
      }
    }
    
    return 0;
}

float boundingBoxes_IoU(vector<Rect> predBox, vector<Rect> gTBox){
  
  // coordinates for intersection rectangle
  int xA = max(predBox[0], gTBox[0]);
  int yA = max(predBox[1], gTBox[1]);
  int xB = min(predBox[0]+predBox[2], gTBox[0]+gTBox[2]);
  int yB = min(predBox[1]+predBox[3], gTBox[1]+gTBox[3]);
  
  float interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1);
  
  // Area of the predicted bounding box and the ground truth
  float predBoxArea = (predBox[0]+predBox[2]) - predBox[0] + 1) * ((predBox[1]+predBox[3]) - predBox[1] + 1);
	float gTBoxArea = ((gTBox[0]+gTBox[2]) - gTBox[0] + 1) * ((gTBox[1]+gTBox[3]) - gTBox[1] + 1);
	
  // compute iou as the intersection of the two boxes divided by their union
  float iou = interArea / float(predBoxArea + gTBoxArea - interArea);
  
	# return the intersection over union value
	return iou;  
}