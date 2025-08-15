# Udacity-CNN-self-driving-car
An end-to-end self-driving car project developed for the Udacity simulator (Udacity_self_driving_car_nanodegree_program) using Convolutional Neural Networks (CNNs). The model predicts steering angles—and in extended versions, throttle—from only raw front-facing camera images.  
  
This project focuses on the experimental deployment of a convolutional neural network (CNN)-based autonomous vehicle in the Udacity Self-Driving Car Simulator. The whole development process includes data collection through manual driving, preprocessing and data augmentation of gathered images, designing and training multiple CNN models, and integrating the learned models into the simulator in auto mode. In total, four models were developed: three specialized CNN models hand-tailored from scratch, and one additional model set up by fine-tuning a pre-trained network on ResNet50. The models learned to predict the steering angle from images received from front-facing cameras and were tested on both Track 1 and Track 2 of the simulator. The best-performing model demonstrated clean, collision-free driving on both tracks, and the remaining models passed Track 1 but failed at a specific declining turn on Track 2. Comparative study of the models was also part of this project in an effort to justify the ultimate selection based on performance and theoretical explanation.
## Data Collection and Preprocessing
### Data Collection
- Data was collected by manually driving the car in training mode on both simulator tracks.
- Track 1 focused on straight paths and slight turns; Track 2 focused on sharp turns and slopes.
- The goal during driving was to stay on the drivable path, not necessarily to follow lane discipline.
- The simulator generated:
  * 320×160 images (left, center, right)
  * A CSV log with
    - image paths
    - steering angle
    - throttle
    - brake
    - speed.
  
<img width="50%" height="511" alt="image" src="https://github.com/user-attachments/assets/6c4a198a-b7b8-44d8-aa10-4a72bb1aefaf" />  

- Around 45 minutes of driving per track resulted in 92,891 data entries.
- A manual review was conducted to remove off-road driving data, ensuring the dataset reflects clean, on-track behavior for better model training
### Preprocessing
- A custom image loading function was implemented to handle image import and transformation.
- Images were converted from BGR to RGB color space for compatibility with deep learning frameworks like PyTorch.
- Whitespace in file paths was removed to avoid image loading errors.
- Images were cropped to focus on relevant driving features:
    * 60 pixels were removed from the top (to discard sky).
    * 25 pixels were removed from the bottom (to discard the car hood)

<img width="50%" height="454" alt="image" src="https://github.com/user-attachments/assets/f8f32284-933b-4bc2-bdcd-9952c9eac482" />

- the image was resized to 200*66 through interpolation  

<img width="50%" height="580" alt="image" src="https://github.com/user-attachments/assets/bbe1ff39-1285-4a1b-808a-a712524c8dd8" />  
  
- To ensure consistent input scaling, all images were normalized to a range of [–1, 1]. This normalization helps stabilize training and is commonly used in CNN pipelines.  

## Data Augmentation  
- A random horizontal flip was applied to ~50% of the input images during training.
- Corresponding steering angles were also flipped (multiplied by -1) to maintain label consistency.
- This helped the model generalize better, especially by balancing left and right turn examples.
- Other augmentations (e.g., brightness, contrast) were intentionally excluded because:
  * The Udacity simulator does not simulate different lighting or weather conditions.
  * Therefore, such transformations were considered irrelevant for this environment.

### Preprocessing and Data Augmentation Strategy

Two main approaches can be used to apply preprocessing and augmentation to datasets:

#### 1. On-the-fly Augmentation
- Applies random transformations in real-time during training
- Generates a new variation of the data on each epoch

#### 2. Offline Augmentation
- Applies transformations before training
- Saves augmented images into an expanded, static dataset

🔷 For this project, on-the-fly augmentation was used.
This method is commonly preferred in deep learning when computational resources are sufficient, which was the case here.

Advantages of On-the-Fly Augmentation:

- Efficient memory and storage usage
- Flexible training – augmentation strategies can be added or modified easily
- Avoids static augmentation that might limit variability

## Models

Several Convolutional Neural Network (CNN) architectures were explored in this project, all inspired by NVIDIA’s end-to-end self-driving car model. Each model shares a common CNN feature extractor, but varies in its fully connected (linear) layers and output strategy.

The objective was to experiment with different architectures that:
- Predict only the steering angle
- Predict both steering angle and throttle
- Predict steering and throttle using a branched dual-output design

### Model 1: Steering-Only Prediction

<img width="975" height="486" alt="image" src="https://github.com/user-attachments/assets/ca07d4af-8839-48b8-aa10-27078c9f0719" />

### Model 2: Dual Output (Steering + Throttle)

<img width="1173" height="565" alt="image" src="https://github.com/user-attachments/assets/5263d8bc-ff0b-412f-86e5-e2975258d720" />

### Model 3: Branched Dual Output

<img width="1152" height="809" alt="image" src="https://github.com/user-attachments/assets/018bcc9b-f2d8-46c2-a8bc-974ea95a68b5" />

