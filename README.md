# CBIS-DDSM-and-MINI-MIAS-clasification-using-pretrained-CNN ❤️

A convolutional neural network (CNN) is a type of artificial neural network specifically designed for processing data that has a grid-like structure, such as an image. It is particularly useful for image classification, object detection, and image generation tasks.

CNNs are composed of several layers of interconnected nodes, with each layer performing a specific operation on the input data. The first layer is the input layer, which takes in the raw image data. The next layer is the convolutional layer, which applies a set of filters to the input data to extract features. These features are then passed through a pooling layer, which reduces the dimensionality of the data by summarizing the extracted features in a fixed-size window. This process is repeated with multiple convolutional and pooling layers until the final layers, which are fully connected and classify the input data into one of the predefined categories.

This repo contains a concise Tensorflow implementation using CNN's models for classifying mammograms. This repository use the following methodology: 

1. For this research, we compared two well-known deep learning models
(ResNet50 and EfficientB7).
2. We enhanced the quality of each cropped image. We applied CLAHE (Con-
trast Limited Adaptive Histogram Equalization), unsharp masking, and a
median filter.
3. We used a data augmentation algorithm from the albumentation library.
For increase the number of training images and have a more robustness
CNN.
4. We compared the performance of each model with five metrics(accuracy,
precision, recall, f1-score, and confusion matrix)


## Setup

To create a virtual environment with TensorFlow using Anaconda, follow these steps:

Open the Anaconda Prompt by clicking the Start button and typing "Anaconda Prompt".
Type the following command to create a new virtual environment called "tfenv":

```python
conda create --name tfenv
```

Activate the virtual environment by typing:

```python
conda activate tfenv
```

Finally, install requirements.txt.

```python
conda install requirements.txt
```

## GPU setup

CUDA is a parallel computing platform and programming model developed by NVIDIA for general computing on graphics processing units (GPUs), and cuDNN is a library developed by NVIDIA for deep learning applications. Here are the steps to install both CUDA and cuDNN on your system:

- Check the version of [TensorFlow](https://www.tensorflow.org/install/source#gpu) that you can use.
- You need to install Visual Studio; the documentation explains which version of VS you need. The one we used was [Visual Studio 2019.](https://my.visualstudio.com/Downloads?q=Visual%20Studio%202019)
- Check if your system has an NVIDIA GPU and if it is compatible with CUDA. You can check the list of CUDA-compatible GPUs at the NVIDIA website.
- Download the CUDA Toolkit and cuDNN from the NVIDIA website. You will need to create an NVIDIA developer account to access the downloads.

The following document gives every link used for this process.

```bash
Install tensorflow GPU.txt
```
## Connect with me

- Contact me CesareduardoMucha@hotmail.com
- Follow me on [Linkedin](https://www.linkedin.com/in/cesar-eduardo-mu%C3%B1oz-chavez-a00674186/) and [Twitter](https://twitter.com/CesarEd43166481) 💡

## Co-authors

- Dr. Hermilo Sanchez Cruz
- Dr. Juan Humberto Sossa Azuela
- Dr. Julio Cesar Ponce Gallegos

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)