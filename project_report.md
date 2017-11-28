# On-demand Image Inpainting using CNN and conditional GAN

### Bowen Shen (bowenshe@usc.edu)

In this blog, we will first talk about why we need image restoration and why we focus on the specific image restoration task, namely image inpainting. Then introduce the general methods based on deep learning that we can apply to handle the image restoration, mostly from existing methods. We will talk about our implementation based on the related research. Finally, we will present and analyze our results and conclude our project.

## Problem formation

Images are never perfect. Defects or imperfections could exist anywhere inevitably due to inexperienced photographers, inappropriate preservation methods, or even some deliberate hacking. In order to minimize the cost of imperfect images, we need specific image restoration methods to handle different image problems. For example, we have four different image restoration task dealing with four different problems: image inpainting for missing block regions, pixel interpolation for missing non-contiguous deleted pixels, image deblurring for blur effects, image denoising for unexpected noise.

In the past, image restoration tasks are usually handled manually. Artists can draw the missing part of the corrupted images based on their imagination. Besides, photographers can process the original photos with Photoshop or other tools to remove noise or blur effects. However, limitations are roots in the traditional manual restoration methods. Human, even an expert, would probably contribute huge amount of time and efforts restore a corrupted image, let alone dozens of ones. Besides, human can’t see all hidden details in the images. In other words, they will probably miss some parts when dealing with whole corrupted images. Those unavoidable limitations prevent the high efficiency and accuracy of image restoration.

Nowadays, deep learning has swept the field of computer vision and produced excellent results on recognition problems, which provides us a new and better method to accomplish image restoration task. In fact, deep learning outperforms the traditional manual method in many ways. First, manual restoration usually contains subject experience on the corrupted images, while deep learning is more general considering the size of training dataset. Second, deep learning can extract more details in the image so as to get better restoration result.

In our project, we will implement deep learning model to conduct specific image restoration task. Given corrupted images with random missing blocks, we will try to refill the missing region and receive relatively perfect images. We choose image inpainting because images get rarely total corruption, mostly in specific part. Once we learn image inpainting, we can easily remove and refill the original corrupted region. In other words, image inpainting is kind of a universal approach in image restoration, which makes it popular in the related research. 

## Related work

In this project, we handle image inpainting using two different deep learning models: CNN and GAN.

First, for convolutional neural networks, there is usually an encoder-decoder structure as follows. Encoder extracts the features from original images while decoder reconstructs images based on the features extracted. Usually, a channel-wise fully connected layer [1], instead of fully connected layer, will connect encoder and decoder, to reduce number of parameters hence the computation and cost.

Original image inpainting usually focuses on images with same sizes of missing content in the center. In reality, image corruption comes with random levels of difficulty: the image can be covered by a mask of arbitrary size or location. In order to consider all possible situation instead of specific difficulty level, on-demand learning method [2] has been proposed to dynamically adjust its focus when needed. Initially, each task is divided into N sub-tasks of increasing difficulty and trained jointly to accommodate all N subtasks. As more difficult subtasks require more training, number of training examples for subtasks will be updated based on PSNR (Peak Signal-to-Noise Ratio), a number to illustrate the correctness of prediction image. A lower PSNR indicates a difficult task (large size of missing), and needs more training samples. Specifically, for different ….

For GAN, there are quite different methods. Context encoder applies adversarial loss associated with original l2 loss to enhance the quality of results.

Besides, image restoration is actually an image-image task. Given imperfect images, we wish to reconstruct the images without knowing what’s missing. Conditional GAN [3] provides a universal approach when dealing with image-image translation. Inputs to the Conditional GAN are actual images instead of noise in original GAN paper [5]. Also, U-Net [34] used in generator can provide quite clear images and Patch GAN for discriminator penalizes structure at the scale of image patches.

## Project implementation

As mentioned before, this project implements two different deep learning models to handle image restoration: CNN and GAN. On-demand learning method is applied to both models to generalize to all difficulty levels. Before implementation of deep learning models, a quite time-consuming job requires preparation, that is data retrieval and preprocessing. 

### data preprocessing

Dataset used in this project is CelebA [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html], which includes 202,599 human face images (jpg). Several tools and libraries are needed to preprocessing jpg files downloaded from google drive (https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8). First, Image from PIL can transform jpg images into numpy arrays, which can be directly used later. Second, misc from scipy can resize the original images into given size. Here, images’ size is 128 * 128 facilitating computation during model formation. Third, how to push all data into training model? It would be memory error as well as if all the data processed concurrently. The solution is to handle data when needed, treating data in batches. Besides, pickle all the paths of images and get data through the path when needed. For the full implementation, refer to datapreprocessing.py ().

### CNN model implementation

Original model is implemented in lua and torch. This project, however, is fully implemented in python and Tensorflow, based on my own understanding of the algorithm in the paper. (I know nothing about lua and torch anyway)

#### Network Architecture
We follow the symmetric encoder-decoder pipeline used in the paper. The encoder takes images of size 128 * 128 as input, and encodes them through five convolutional layers associated with Batch normalization layer and leaky-relu layer. The decoder takes representation features and generates restored images through five conv_transpose layers with batch-normalization layer and relu layer. The last layer is followed by tanh activation function instead of relu function. A channel wise fully connected layer connects encoder and decoder.

#### On-demand learning methods
Originally, for image inpainting task, we remove central block of fixed size to simulate the corrupted image, the input to the image-inpainting model. For on-demand learning, however, we need to remove any random part of the images of any arbitrary size. In other words, to get the input for our on-demand model, we need to crop out any possible part (block for simplicity). We define five size ranges of covering mask to simulate different difficulty levels: 1-18, 19-36, 37-54, 55-72, 73-90. We choose ninety as the maximum size of mask because images will become unreasonable and impossible to recover once losing too much. 

To begin with, each sub task has 18 samples per batch (batch size = 90). After each iteration, we update the number of each sub task using equation mentioned before, and divide new batch with the new subtask number. We can imagine that after a few iterations, most difficult subtask 5 would cover most samples within a batch as more training data are needed for more difficult job. This on-demand learning applies to GAN version as well.

#### Parameter choice
We set 90 as batch size, 15 epochs for training, 5 difficulty levels, 98-1-1 train-valid-test division. 

### GAN model implementation
Conditional GAN originally works on image-to-image translation, like image colorization or …But here, we focus on image inpainting using conditional GAN. The generator consists of seven encoders and seven decoders. Just like CNN model, both encoder and decoder are composed of convolutional layer, batch normalization layer, and relu layer. The difference lies on symmetric skip connections in conditional GAN. For instance, the last layer takes the output of the first layer as input as well as its previous layer. By skipping middle layers, many details of original images are shared in the very last layer, thus generating high resolution images. We can see the difference from the following figure. The discriminator uses Patch-GAN which only penalizes structures at the scale of patches. Each N*N patch is classified as real or fake across the image, giving the sense of texture or style loss.

<p align="center">
  <img width="400" src ="https://github.com/skylinebreaker/On-Demand-Image-Inpainting/blob/master/images/U-Net.png"/>
</p>

Following the same rule of CNN model, we add on-demand learning method to conditional GAN, making the model tolerable for any level of difficulty.

#### Parmeter choice
We set 90 as batch size, 20 epochs for training, 5 difficulty levels, 98-1-1 train-valid-test division. 

### My trials
Besides the traditional Conditional GAN method, we try to change to generator loss function. Originally loss function contains a L1 term to illustrate L1 distance between the real and fake images, which speeds up the training process to converge to the more real images. I change a little bit, adding one more term to penalize more on the missing part of GAN.

## Project results
We implement all the models on Google Cloud Platform and almost run out of all the free credits ($600) of two accounts.

### CNN model
To verify the advantage of on-demand learning, we train two different model for comparison. First model is just the one mentioned before, using on-demand learning methods, and the second model only trains on specific difficulty (block mask of size 90). Both train for 15 epochs, and share same parameters except whether apply on-demand methods. We can see loss changes through the following figure.

<p align="center">
  <img src ="https://github.com/skylinebreaker/On-Demand-Image-Inpainting/blob/master/images/CNN_loss.jpeg"/>
</p>

We also show some images generated from two models.

<p align="center">
  <img src ="https://github.com/skylinebreaker/On-Demand-Image-Inpainting/blob/master/images/CNN_results.jpeg"/>
</p>

Based on two figures above, we find that on-demand method can reduce the loss when dealing with spectrum of difficulty level, while fixated approach usually fails at other difficulty level. We also test other situation, like with same number for all difficulty level, or median difficulty fixated level, and receive similar results. Therefore, we can make sure on-demand learning methods does apply to handle realistic image inpainting across all difficulty levels.

### GAN model
It’s kind of tricky to train GAN model on google cloud, and sometimes hard to explain what we get from GAN. We train our version of conditional GAN with on-demand methods on the google cloud. And we get following results.

<p align="center">
  <img src ="https://github.com/skylinebreaker/On-Demand-Image-Inpainting/blob/master/images/GAN_loss.jpeg"/>
</p>

<p align="center">
  <img src ="https://github.com/skylinebreaker/On-Demand-Image-Inpainting/blob/master/images/GAN_results.jpeg"/>
</p>

Compare with CNN model, the whole image seems sharper except for the missing region, which can be explained by the U-Net. However, the refillment of missing region vary dramatically. In general, it works well at simple task and bad at complicated tasks. Especially when the background is black, most of the content in the missing region is also black. And the border of missing region is more obvious in GAN model regardless of tasks. Pure conditional GAN seems not a good choice of image inpainting. To make it better fit our task, I try several ways to modify the model. 

### My trials
My very first trial is to remove U-Net. As we know from the U-Net, the fist image, containing quite number of missing region would greatly affect the final images generated, especially for difficult tasks. As a result, the output images will keep those black points from the input images, thus causing dark missing refillment. So, I remove U-Net. However, the images generated barely follow the same type of the input images and more distorted.

So, I have to keep the U-Net and try to modify the loss function as mentioned before. For the early epochs, the modification achieves less loss and higher PSNR. However, as time goes, the modification becomes the same or even worse than traditional conditional GAN. This can be explained that the purpose of GAN is just to let the model to decide the best performed loss function without human intervention. Adding the new term to generator would cause additional burden to GAN, thus harming the performance in the long term.

Both of my trials fail and that won’t stop me. For the future work, I wish to update the loss function for discriminator. The task here actually differs from original purpose of conditional GAN. We don’t need to worry about the input X here. We should pay more attention to the missing part. I still thinking about any potential optimization that could work indeed.

## Summary
