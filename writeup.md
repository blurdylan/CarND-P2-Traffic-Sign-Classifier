# **Traffic Sign Recognition** 

## Writeup Template

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image4]: ./images/processed/output/speed_limit_30.jpg "Traffic Sign 1"
[image5]: ./images/processed/output/8.jpg "Traffic Sign 2"
[image6]: ./images/processed/output/drive_right.jpg "Traffic Sign 3"
[image7]: ./images/processed/output/no_stop.jpg "Traffic Sign 4"
[image8]: ./images/processed/output/school.jpg "Traffic Sign 5"
[image9]: ./images/processed/output/slippery.jpg "Traffic Sign 6"

## Rubric Points
* **File Submission**
* **Dataset Exploration**
* **Design and Test a Model Architecture**
* **Test Model on New Images**

### Writeup / README
You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### File Submission:
 The files needed to be submitted were;
  * This writeup
  * The jupyter notebook associated
### Dataset Exploration: 
The dataset was summarized and shown before preprocessing, after preprocessing and after augmentation, a visualization was also done to show distribution of images per class.

* The data was counted mostly using the python `len()` function.
* 16 images picked in 6 (randomly choosen) classes are first displayed.
* A bar chart is drawn from the data counted in each classes and the minimum, max and average number of images per class is computed.

#### Results
Number of training examples = 34799

Number of validation examples = 4410

Number of testing examples = 12630

Image data shape = (32, 32, 3)

Number of classes = 43

![preprocessed Image grid](./images/writeup/pre_image_grid.png)

![Dataset chart](./images/writeup/dataset_chart.png)


### Design and Test a Model Architecture : 

#### 1. Pre processing
 * The class images were normalized and grayscaled using the basic statistic tools, to ease the training and create uniformity through out the classes.

  ![processed Image grid](./images/writeup/pro_image_grid.png)

  * The data was augmented through distortion and the augmented data was stored, this is done to help the training.
  
  ![Undistorted](./images/writeup/undistorted.png)
  ![Undistorted](./images/writeup/distorted.png)

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| Input = 28x28x32. Output = 14x14x32.
 				|

 


#### 3. How Was The Model Trained

  * The LeNet-5 CNN from the course was implemented with the hyperparameters `mu = 0` and `sigma = 0.1`.
  * The initial `learning_rate = 0.001`, the `batch_size = 128 image`, model was trained for `12 epochs`.
  * TensorFlow design functions were used to train, validate and test the model architecture, training was done through the Adam optimizer.

#### 4. Approach Taken

My final model results were:
* training set accuracy of 99.8
* validation set accuracy of 98.4
* test set accuracy of 97.2

If a well known architecture was chosen:
* LeNet-5 CNN architecture was taken
* As it is an image classification problem, convolutional layers were used as it is common in modern image classification systems as reduce computation (as compared to classical neural network with only fc layers). It seems to me that inception moduls are essential for good performance on such kind of tasks as they allow to do not select optimal layer (say, convolution 5x5 or 3x3), by perform different layer types simultaneously and it selects the best one on its own.
* The model is set to be saved on tf if and only if `validation_accuracy > 96%`
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit (30km/h)     		| Speed Limit (30km/h)  									| 
| Speed Limit (60km/h)     			| Speed Limit (60km/h) 										|
| Keep Right					| Keep Right											|
No Vehicles	      		| No Vehicles					 				|
| Children Crossing			| Children Crossing      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

You can find these information in the [html notebook](./Traffic_Sign_Classifier.html) in the [source code](./)

For the first image, the model is 100% sure that this is a 30kmph speed limit, The top five soft max probabilities were

----------
1.0000 Speed limit (30km/h)

0.0000 Speed limit (20km/h)

0.0000 Speed limit (50km/h)

0.0000 Speed limit (70km/h)

0.0000 Speed limit (80km/h)

----------
For the second image, The top five soft max probabilities were:

----------
0.9343 Speed limit (60km/h)

0.0654 Speed limit (80km/h)

0.0003 Speed limit (50km/h)

0.0000 Speed limit (30km/h)

0.0000 Speed limit (120km/h)

----------
For the third image, The top five soft max probabilities were:

----------
1.0000 Keep right

0.0000 Turn left ahead

0.0000 Priority road

0.0000 Beware of ice/snow

0.0000 Speed limit (80km/h)

----------
For the fourth image, The top five soft max probabilities were:

----------
0.4049 No vehicles

0.1455 Keep left

0.0929 Speed limit (70km/h)

0.0783 No passing

0.0466 Turn right ahead

----------
For the fifth image, The top five soft max probabilities were:

----------
1.0000 Children crossing

0.0000 Bicycles crossing

0.0000 Right-of-way at the next intersection

0.0000 Dangerous curve to the right

0.0000 Beware of ice/snow
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

