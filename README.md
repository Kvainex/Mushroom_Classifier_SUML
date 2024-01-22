# Mushroom Classification SUML

An image classification project which detects different kinds of mushroom species using **Keras** and **Tensorflow**.  Image Classification is a Machine Learning module that trains itself from an existing dataset of multiclass images and develops a model for future prediction of similar images not encountered during training. Developed using Convolutional Neural Network (CNN).

Tensorflow: https://www.tensorflow.org/

Keras: https://keras.io/

## Convolutional Neural Network (CNN)

A convolutional neural network (CNN) is a type of artificial neural network used in image recognition and processing that is specifically designed to process pixel data.

More information can be found here: https://www.analyticsvidhya.com/blog/2021/06/image-processing-using-cnn-a-beginners-guide/


## Pre-requisites

Create a python virtual environment using:
```bash
python -m venv /path/to/new/virtual/environment
```

Then, fork the code and place it in the root folder.

Next, activate the virtual environment using (Windows):
```bash
Scripts/activate
```

Check https://docs.python.org/3/library/venv.html for more details.

After the virtual environment is activated, install the dependencies needed using:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset that was used is a kaggle dataset https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images

In total there were:
| Datasets | Number of Images |
| ----------- | ----------- |
| Training Set | 5372 |
| Validation Set | 1342 |
| Total | 6714 |

80% of the dataset was used for the Training Set and the 20% for the Validation Set.

## Experiments

### Normalization
For normalization, the image's were rescaled. RGB values from [0, 255] to [0, 1] using:
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling

### Data Augmentation
I augmented the data using:
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomFlip
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomRotation
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomZoom

Which basically will randomly **rotate**, **flip**, and **zoom** the images when training.

## Results

Here is the result for the **ResNet50** model:

![ResNet50](https://github.com/Kvainex/Mushroom_Classifier_SUML/blob/main/Training_result.png)

![Graphs](https://github.com/Kvainex/Mushroom_Classifier_SUML/blob/main/Training_graphs.png)