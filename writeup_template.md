## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2.1]: ./output_images/HOG_example.png
[image2.2]: ./output_images/HOG_example2.png
[image3.1]: ./output_images/sliding_windows.png
[image3.2]: ./output_images/sliding_windows2.png
[image3.3]: ./output_images/sliding_windows3.png
[image4]: ./output_images/sliding_window.png
[image5.1a]: ./output_images/image0018.png
[image5.1b]: ./output_images/heatmap0018.png
[image5.2a]: ./output_images/image0019.png
[image5.2b]: ./output_images/heatmap0019.png
[image5.3a]: ./output_images/image0020.png
[image5.3b]: ./output_images/heatmap0020.png
[image5.4a]: ./output_images/image0021.png
[image5.4b]: ./output_images/heatmap0021.png
[image5.5a]: ./output_images/image0022.png
[image5.5b]: ./output_images/heatmap0022.png
[image5.6a]: ./output_images/image0023.png
[image5.6b]: ./output_images/heatmap0023.png
[image6]: ./output_images/finalhm0023.png
[image7]: ./output_images/outimg0023.png
[video1]: ./project.mp4

## [Rubric Points](https://review.udacity.com/#!/rubrics/513/view)
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images to get a hint of the quality and characteristics of the images. This is an example of a random image from each of the classes:
![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2.1]
![alt text][image2.2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, always balancing accuracy versus CPU performance (very relevant for the video processing step). This is the final choice of HOG parameters:
- Orientations (8): bigger orientations did not lead to significant accuracy improvement
- Pixels per cell (8x8): smaller values did not lead to significant accuracy improvement
- Cells per block (2x2): smaller values did not lead to significant accuracy improvement
- Color space (YCrCb): notably better accuracy than RGB, and similar accuracy as YUV and HSV.
- Color channels (ALL): all color channels was clearly the winner in terms of accuracy over selecting any other single channel

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in the fourth cell of the IPython notebook. I first tested with HOG features only, with an accuracy of 98.6%. By adding spatial features and color histogram features I could increase the accuracy up to 99.4%.  

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

This is implemented in the 5th cell of the IPython notebook. I decided to search within the test images using 120x120 windows with 0.8 overlap, obtaining the following results (only 2 samples shown):
![alt text][image3.1]
![alt text][image3.2]

However, small cars were not properly detected, so another search with a smaller window size (80x80) was required to identify them:
![alt text][image3.3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales (0.75, 1, 1.5, 2, 3) using HSV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  I also used a decision threshold to ensure the car predictions are made with high confidence, to avoid false positives. In other words using SVM terminology, predictions too close to the decision boundary will not be considered. Here are some example images with the windows found for the aforementioned scales:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

This is implemented in the sixth code cell of the IPython notebook. I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here are six frames and their corresponding heatmaps:

![alt text][image5.1a]![alt text][image5.1b]
![alt text][image5.2a]![alt text][image5.2b]
![alt text][image5.3a]![alt text][image5.3b]
![alt text][image5.4a]![alt text][image5.4b]
![alt text][image5.5a]![alt text][image5.5b]
![alt text][image5.6a]![alt text][image5.6b]

Here is the output of the integrated heatmap from all six frames:

![alt text][image6]

Here the resulting bounding boxes are drawn onto the last frame in the series after applying `scipy.ndimage.measurements.label()` to the integrated heatmap:

![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found this approach very unreliable and CPU intensive. After studying deep learning, I believe a more robust car detection technique should be possible with a neural network by looking at the pixels directly. It would also be much simpler as it should not require any image processing nor sliding window search algorithms.

As in all projects of term 1, the biggest challenge has been finding out the best hyper-parameters for the model. As we cannot explore the whole combination of hyper-parameters, the fine-tuning process requires a lot of experience and intuition.

I believe my pipeline would fail with different lightning and weather conditions.

To improve the robustness of my model, next steps would be:
1. Using a dynamic y_start_stop value based on the given scale
2. Train with more data and use a test set that is not related to the training set
