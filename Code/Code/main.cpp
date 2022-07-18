#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

using namespace cv;

int main(int argc, char** argv)
{
    //Load Yolo Model
    cv::dnn::Net net = cv::dnn::readNet("../../../Model/yolov3_training_last.weights", "../../../Model/yolov3_testing.cfg");
    
    String classes = ["Hands"];
    
    // Initialize the parameters
    float confThreshold = 0.5; // Confidence threshold
    float nmsThreshold = 0.4;  // Non-maximum suppression threshold
    int inpWidth = 416;        // Width of network's input image
    int inpHeight = 416;       // Height of network's input image
    
    // gather images in a specified path
    vector<String> images_path;
    glob("Dataset progetto CV - Hand detection _ segmentation/rgb/*.jpg", images_path, false);  
    
    vector<String> output_layers = getOutputNames(net);
    
    randShuffle(images_path);
    
    
    // process images for Yolo neural network
    Mat img, blob;
    
    for (int i = 0; i < images_path.size; i++){
      img = imread(images_path(i));
      
      int img_width = img.cols;
      int img_height = img.rows;
      
      blobFromImage(img, blob, 1/255, cv::Size(img_width, img_height), Scalar(0,0,0), true, false);
      
      net.setInput(blob);
      
      vector<Mat> outs;
      net.forward(outs, output_layers);
      
      //
      postprocess(img, outs);
      
    }
    
    return 0;
}


void postprocess(Mat& frame, const vector<Mat>& outs){

    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;    

    for (size_t i = 0; i < outs.size(); ++i){

    // Scan through all the bounding boxes output from the network and keep only the
    // ones with high confidence scores. Assign the box's class label as the class
    // with the highest score for the box.

    float* data = (float*)outs[i].data;

    for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols){
        
        Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
        Point classIdPoint;
        double confidence;

        // Get the value and location of the maximum score

        minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

        if (confidence > confThreshold){

            int centerX = (int)(data[0] * frame.cols);
            int centerY = (int)(data[1] * frame.rows);
            int width = (int)(data[2] * frame.cols);
            int height = (int)(data[3] * frame.rows);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            classIds.push_back(classIdPoint.x);

            confidences.push_back((float)confidence);

            boxes.push_back(Rect(left, top, width, height));

            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences
    vector<int> indices;
    
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
    }
}


// Get the names of the output layers
vector<String> getOutputsNames(const Net& net){
      
  static vector<String> names;
  if (names.empty()){
    //Get the indices of the output layers, i.e. the layers with unconnected outputs
    vector<int> outLayers = net.getUnconnectedOutLayers();
    
    //get the names of all the layers in the network
    vector<String> layersNames = net.getLayerNames();
    
    // Get the names of the output layers in names
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i)
    names[i] = layersNames[outLayers[i] - 1];
    }
  return names;
}