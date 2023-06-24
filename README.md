# Human hands detection and segmentation 

**Computer Vision course - Summer project 2022**

The goal of this project is to develop a system capable of:
- detecting human hands in an input image;
- segmenting them from background.

## Benchmark Dataset

It contains a total of 30 images, from EgoHands and HandOverFace datasets, categorized by level of difficulty:
1) images with similar backgrounds where few hands are present and clearly visible;
2) images with different backgrounds and many hands present, possibly with partial occlusions;
3) general hand pictures of people of different skin tone and gender.

## Training and Validation Datasets

Considering EgoHands (4800 images) and HandOverFace (300 images) datasets:
1) we removed the 30 images corresponding to the given benchmark (test) dataset;
2) we randomly picked 50 images for the validation set (40 from EgoHands and 10 from HandOverFace);
3) we merged all other remaining images into a single training dataset;
4) we added 20 additional personal images (with ground truth manually drawn using https://github.com/heartexlabs/labelImg) in the validation set.

Final training set: 5020 images (with corresponding 5020 labels).<br />
Final validation set: 70 images (with corresponding 70 labels).<br />
&rarr; cv-dataset.zip is the full custom dataset used for training and validation (not provided in this repository).

Additional images added in the validation set to be sure to get the best output from the neural network. So that it generalizes hand detection in the best way, avoiding overfitting.

## Group members

Simone Bastasin, Alessandro Lotta, Elisa Varotto.