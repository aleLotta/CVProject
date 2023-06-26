#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include "segmentation.h"
using namespace cv;
using namespace cv::dnn;
using namespace std;


int main()
{
    Mat input_image, bboxes_image, gt_mask, my_mask, segmented_image;
    vector<Mat> detections;
    vector<Rect> gt_boxes, pred_boxes;
    int x, y, w, h;
	float iou, pixel_acc;
	String savingPath;
	
    // Load paths of images, labels, masks and results
    vector<string> image_paths, label_paths, mask_paths;
    //glob("../Dataset progetto CV - Hand detection _ segmentation/rgb/*.jpg", image_paths, false); // 30 imaes
    //glob("../Dataset progetto CV - Hand detection _ segmentation/det/*.txt", label_paths, false); // 30 labels
    //glob("../Dataset progetto CV - Hand detection _ segmentation/mask/*.png", mask_paths, false); // 30 masks
    glob("../benchmark_dataset/rgb/*.jpg", image_paths, false); // 30 imaes
    glob("../benchmark_dataset/det/*.txt", label_paths, false); // 30 labels
    glob("../benchmark_dataset/mask/*.png", mask_paths, false); // 30 masks
    ofstream output_file = ofstream("../metric_results.txt");

    // Load model
    Net net = readNet("../Model/yolov5m.onnx");

    /* Process images and labels */
	
    for (int i = 0; i < image_paths.size(); i++)
    {
        gt_boxes.clear();
		
		// Load image
        input_image = imread(image_paths[i]);
		
        // Load ground truth boxes (from labels)
		ifstream input_file = ifstream(label_paths[i]);
        while (input_file >> x >> y >> w >> h)
            gt_boxes.push_back(Rect(x, y, w, h));
        input_file.close();
		
		// Detect bounding boxes using neural network
        detections = PreProcess(input_image, net);
		
		// Process detected bounding boxes to keep correct boxes
        input_image.copyTo(bboxes_image);
        pred_boxes = PostProcess(bboxes_image, detections);
		
        //imshow("Detected Image" + to_string(i + 1), bboxes_image);
        //waitKey(0);
        savingPath = "../Detection/" + to_string(i + 1) + ".jpg";
        imwrite(savingPath, bboxes_image);
		
		// Calculate image IoU metric
        iou = ImageIouMetric(gt_boxes, pred_boxes);

        // Segmentation
        gt_mask = imread(mask_paths[i], IMREAD_GRAYSCALE);
        my_mask = Mat::zeros(input_image.rows, input_image.cols, CV_8U);
        segmented_image = HandSegmentation(input_image, my_mask, pred_boxes);

        //imshow("Segmented Image" + to_string(i + 1), segmented_image);
        //waitKey(0);
        savingPath = "../Segmentation/" + to_string(i + 1) + ".jpg";
        imwrite(savingPath, segmented_image);

        // Caluclate pixel accuracy
        pixel_acc = PixelAccuracy(gt_mask, my_mask);

		output_file << "Image " + to_string(i + 1) + "\n";
		output_file << "IoU = " + to_string(iou) + "\n";
		output_file << "PixAcc = " + to_string(pixel_acc) + "\n\n";
    }
	output_file.close();
	
    return 0;
}
