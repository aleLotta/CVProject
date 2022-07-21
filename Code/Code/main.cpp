/*import numpy as np

net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg')

classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

#cap = cv2.VideoCapture('video4.mp4')
#cap = 'test_images/<your_test_image>.jpg'
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    #_, img = cap.read()
    #img = cv2.imread("test_images/<your_test_image>.jpg")
    img = cv2.imread("30.jpg")
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 1, (255,255,255), 2)

    #cv2.imshow('Image', img)
    cv2.imwrite("Image.jpg", img)
    break
#cap.release()
cv2.destroyAllWindows()*/


#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <typeinfo>

using namespace cv;
using namespace std;
using namespace cv::dnn;

vector<String> getOutputsNames(const Net& net);

int main(int argc, char** argv)
{
    //Load Yolo Model
    cv::dnn::Net net = cv::dnn::readNet("../../../Model/yolov3_training_last.weights", "../../../Model/yolov3_testing.cfg", "Darknet");
    //auto net = cv::dnn::readNet("yolov3_training_last.weights", "yolov3_testing.cfg", "Darknet");
    //cv::dnn::Net net = cv::dnn::readNetFromDarknet("yolov3.cfg", "yolov3.weights");
    
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    
    String classes = {"Hands"};
    
    Mat img = imread("../../Dataset progetto CV - Hand detection _ segmentation/rgb/01.jpg");
    
    
    Mat blob = blobFromImage(img, 1/255, cv::Size(416,416), Scalar(0,0,0), true, false);
    //Mat blob = blobFromImage(img, 0.01, Size(224, 224), Scalar(104, 117, 123));
    //blobFromImage(img, blob, 1/255, cv::Size(416,416), Scalar(0,0,0), true, false);
    
    
    net.setInput(blob);
    
    vector<String> output_layers_names = getOutputsNames(net);
    
    vector<Mat> outs;
    
    net.forward(outs, output_layers_names);
    
    vector<Rect> boxes;
    vector<float> confidences;
    vector<int> classIds;
    
    Mat temp;
    
    
    for (size_t i = 0; i < outs.size(); ++i){
      
      float* data = (float*)outs[i].data;
      for (int j = 0; j<outs[i].rows; ++j){
        //cout<<typeid(data).name()<<endl;
        
        //Mat detection = outs[i].row(j);
        Mat detection = outs[i];
        //cout<<detection<<"**"<<endl;
        
        //Mat scores = detection.col(5);
        Point classId;
        float confidence = detection.at<float>(j,5);        
        //cout<<confidence<<endl;
        
        //minMaxLoc(scores, NULL, &confidence, NULL, &classId);
        if (confidence > 0){
          cout<<confidence<<endl;
        }
        
        //cout<<detection.col(5)<<endl;
        temp.push_back(detection.col(5));
        
        /*if (detection.col(5)Mat::zeros(1,1,CV_64F)) {
          cout<<detection.col(5)<<endl;
        }*/
      }
    }
    cout<<sum(temp)<<endl;
    
    return 0;
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
