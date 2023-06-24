#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

#include "segmentation.h"
using namespace cv;
using namespace cv::dnn;
using namespace std;


/*** Global Parameters ***/

// costants
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;
// colors
Scalar BLACK = Scalar(0, 0, 0);
Scalar RED = Scalar(0, 0, 255);
Scalar GREEN = Scalar(0, 255, 0);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar BLUE = Scalar(255, 178, 50);
vector<Scalar> colors = {GREEN, BLUE, RED, YELLOW};


/*** Pre-Processing ***/

vector<Mat> PreProcess(const Mat& image, Net& net)
{
    Mat blob;
    vector<Mat> outputs;
	
	// Convert to blob
    blobFromImage(image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);
    // Forward propagate
    net.setInput(blob);
	net.forward(outputs, net.getUnconnectedOutLayersNames());
	
    return outputs;
}


/*** Post-Processing ***/

vector<Rect> PostProcess(Mat& image, const vector<Mat>& outputs)
{
    // Only 1 class id (0: hand)
    Point class_id;
    class_id.x = 0;
    class_id.y = 0;
    // Initialize vectors to hold respective outputs while unwrapping detections
    vector<float> confidences;
    vector<Rect> boxes, pred_boxes;
    // Resizing factor
    float x_factor = image.cols / INPUT_WIDTH;
    float y_factor = image.rows / INPUT_HEIGHT;
    float* data = (float*)outputs[0].data;
    const int dimensions = 6;
    const int rows = 25200; // 25200 for default size 640
    vector<int> indices;

    /* Filter Good Detections */

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

    /* Discard Overlapping Boxes */

    // Perform Non-Maximum Suppression and draw predictions
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Save detected Box
        pred_boxes.push_back(Rect(left, top, width, height));
        // Draw bounding box
        rectangle(image, Point(left, top), Point(left + width, top + height), colors[i%colors.size()], 2);
    }
	
	return pred_boxes;
}


/*** IoU Metric ***/

float StandardIouMetric(const Rect& gt_box, const Rect& pred_box)
{
    // Coordinates of the intersection rectangle
    int xA = max(gt_box.x, pred_box.x);
    int yA = max(gt_box.y, pred_box.y);
    int xB = min(gt_box.x + gt_box.width, pred_box.x + pred_box.width);
    int yB = min(gt_box.y + gt_box.height, pred_box.y + pred_box.height);
	
    // Area of the intersection rectangle
    float interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1);
	
    // IoU as the intersection area of the two boxes divided by their union area
    return interArea / float(gt_box.area() + pred_box.area() - interArea);
}

float ImageIouMetric(const vector<Rect>& gt_boxes, const vector<Rect>& pred_boxes)
{
	int nMin = min(gt_boxes.size(), pred_boxes.size());
	int nMax = max(gt_boxes.size(), pred_boxes.size());
	float temp_iou, best_iou, final_iou = 0;
	// Vectors to check if predicted and ground truth boxes have already been evaluated for final iou value
	// initialized with all 0's, i-th element = 1 when its IoU is actually evaulated for final iou value
	vector<int> predEvaluated(pred_boxes.size(), 0);
	vector<int> gtEvaluated(gt_boxes.size(), 0);
	int gtIndex = -1, predIndex = -1;
	
	// Iterating through all possible couples of predicted and ground truth boxes, at each iteration
	// it matches the best couple for remaining boxes and evaluates corresponding IoU metric.
	// This is done until one of the two sets of boxes ends 
	for (int k = 0; k < nMin; k++)
	{
		// To save current best IoU couple to find the new best couple
		best_iou = -1;
		
		for (int i = 0; i < pred_boxes.size(); i++)
		{
			// If current predicted box not considered yet
			if (predEvaluated[i] != 1)
			{
				for (int j = 0; j < gt_boxes.size(); j++)
				{
					// If current ground truth box not considered yet
					if (gtEvaluated[j] != 1)
					{
						temp_iou = StandardIouMetric(pred_boxes[i], gt_boxes[j]);
						
						// If IoU metric of current couple is better than the best one seen so far
						// for remaining boxes: (temporary) update and set boxes as evaluated
						if (temp_iou > best_iou)
						{
							best_iou = temp_iou;
							predIndex = i;
							gtIndex = j;
						}
					}
				}
			}
		}
		// Evaluate considering last temporary update as the best one for this iteration
		predEvaluated[predIndex] = 1;
		gtEvaluated[gtIndex] = 1;
		final_iou += best_iou;
	}
	
	// Full image IoU considering also the mismatching number of boxes between
	// predicted and ground truth boxes, considered all of them with IoU = 0
	return final_iou / nMax;
}


/*** Hand Segmentation ***/

Mat HandSegmentation(const Mat& image, Mat& final_mask, const vector<Rect>& boxes)
{
	Mat final_image, result_mask, bgModel, fgModel;
	image.copyTo(final_image);
	
	// For each predicted box in the image
    for (int k = 0; k < boxes.size(); k++)
	{
		Rect box = boxes[k];
		
        // GrabCut segmentation algorithm for current box
        grabCut(image,		// input image
            result_mask,	// segmentation resulting mask
            box,			// rectangle containing foreground
            bgModel, fgModel,	// models
            3,				// number of iterations
            cv::GC_INIT_WITH_RECT);

        // Generate output image for current box
        Mat foreground(image.size(), CV_8UC3, BLACK);

        // Draw background (black) or hand (white) pixels on foreground image
        for (int i = 0; i < image.rows; i++)
            for (int j = 0; j < image.cols; j++)
                if (result_mask.at<uchar>(i, j) == 0 || result_mask.at<uchar>(i, j) == 2)
				{
                    foreground.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
                    result_mask.at<uchar>(i, j) = 0;
                }
                else
				{
                    foreground.at<Vec3b>(i, j)[0] = colors[k][0];
                    foreground.at<Vec3b>(i, j)[1] = colors[k][1];
                    foreground.at<Vec3b>(i, j)[2] = colors[k][2];
                    result_mask.at<uchar>(i, j) = 255;
                }

        // Computation of the final mask
        final_mask += result_mask;

        // Merge the original image with the mask
        addWeighted(final_image, 1, foreground, 0.5, 0.0, final_image);
    }

    return final_image;
}


/*** Pixel Accuracy Metric ***/

float PixelAccuracy(const Mat& gt_mask, const Mat& my_mask)
{
    float true_neg = 0, true_pos = 0;
	
    for (int i = 0; i < gt_mask.cols; i++)
        for (int j = 0; j < gt_mask.rows; j++)
		{
            // Calculate non hand pixels that are correctly detected
            if (gt_mask.at<Vec3b>(j, i) == Vec3b(0,0,0) && my_mask.at<uchar>(j, i) == 0)
                true_neg++;
			
            // Calculate hand pixels that are correctly detected			
            if (gt_mask.at<Vec3b>(j, i) == Vec3b(255,255,255)  && my_mask.at<uchar>(j, i) == 255)
                true_pos++;
        }
		
    // Pixel accuracy = (TN + TP)/(TP + TN + FP + FN)
    return (true_neg + true_pos) / gt_mask.total();
}
