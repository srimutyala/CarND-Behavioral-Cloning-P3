#**Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* final.mp4 to check how the model performed

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. Pre-procesing

I employed two simple pre-processing techniques on the data. First is using a lambda layer to noralize the data & next is to crop out portion of the image not useful to the training.

####2. Model

My model consists of series of convolution layers with RELU activation. The filter sizes and epths are as below:

1. 1 7x7 layer with a depth of 36
2. 2 5x5 layers with depths of 48 and 64.
3. 2 3x3 layers with depths of 80 each

This is followed by 3 fully connected layers also with RELU activation.

####3. Attempts to reduce overfitting in the model

I did not use a dropout layer for this model as the dat set itself is very small and based on the performance, it did not warrant that. The model was trained multiple times (along with testing of the model) to see how the performance changes by selecting different data for training each time. The performance is not distinguishablly different.

####4. Model parameter tuning

The model used an adam optimizer and mean square error as its loss function. There is a 'steering_correction' parameter that's set to 0.4. This determines how much positive/negative bias needed to be applied to the steering values for corresponding left/right images.

####5. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

It took a while to realize but the quality of the training data made a lot of difference in how well the model performed. Initial runs with the default data did not provide robust performance. Adding the input from the left abd right cameras as well as a parameter for controlling the amount of steering did help a little but still was lacking. I collected some training data driving in the simulator. I also added recovery data by recording he car getting back on the road when it went off it (not recording when it was going off the road but just the recovery). This data set is large & took a while to train but also suffered from driving off the road at certain points. 

A different approach is then taken for data collection where the data set is small but contains mostly recovery information with some normal driving. Surprisingly, this worked better for a few variations of the final model I put together.

My initial thought was to implement the same network I used for the second project (traffis sign classifier). After running into a few roadblocks implementing inception layers, I questioned the wisdom of such a complex network. So, I started with a few convolution layers to test the performance. After surprisngly decent performance with small epoch runs, I decided to just tweak this simple model than implement the model based on the inception and its long training run-times.

I ran into a few scenarios where the vehcile was not providing enough positive/negative steering to keep it on the track. Careful re-do of the training data capture fixed many of those deviants.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consists of 5 convolution neural networks & 4 fully connected networks with the following layers and layer sizes

<INSERT MODEL DETAILS>

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
