# Image Colorization with GANs 
In this project, we tried to approach the image colorization problem by using a conditional generative adversarial network (CGAN) and a Wasserstein generative adversarial network (WGAN). The networks were trained on two public datasets: [Stanford dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) and [VGG flower dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). The colorized results of CGAN and WGAN are [shown here.](#places365-results) For the result evaluation of CGAN, We achieved a 42% perceptual realism in the [turing test.] 

## Prerequisites
- Linux
- Tensorflow 1.7
- Python 3.6
- NVIDIA GPU (V100, 200G memory) + CUDA cuDNN

## Getting Started
### Installation
Clone this repo:
```bash
https://github.com/yxding/ImageColorization.git
cd Colorizing-with-GANs
```

### Dataset
- [Stanford dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [VGG flower dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).
  
  After downloading the data, put them under the `datasets` folder.

### Training
- To train the model on dogs dataset with tuned hyperparameters:
```
python train.py \
  --seed 100 \
  --dataset dogs \
  --dataset-path ./dataset/dogs \
  --checkpoints-path ./checkpoints \
  --batch-size 32 \
  --epochs 200 \
  --log True \
  --lr 3e-4 \
  --lr-decay-steps 1e4 \
  --augment True
  
```

- To train the model of flowers dataset with tuned hyperparameters:
```
python train.py \
  --seed 100 \
  --dataset flowers \
  --dataset-path ./dataset/flowers \
  --checkpoints-path ./checkpoints \
  --batch-size 32 \
  --epochs 200 \
  --log True \
  --lr 3e-4 \
  --lr-decay-steps 1e4 \
  --augment True
  
```

### Evaluate and Sample
- To evaluate the model and sample colorized outputs on the test-set, we first plot the loss and accuracy per epoch and choose the checkpoint which records the best model in the `checkpoints` folder. Then copy that checkpoint to the `test` folder in `checkpoints`.
- Run `test-eval.py` script:
```bash
python test-eval.py
```

## Method

### Generative Adversarial Network
Both generator and discriminator use CNNs. The generator is trained to minimize the probability that the discriminator makes a correct prediction in generated data, while discriminator is trained to maximize the probability of assigning the correct label. This is presented as a single minimax game problem:
<p align='center'>  
  <img src='img/gan.png' />
</p>
In our model, we have redefined the generator's cost function by maximizing the probability of the discriminator being mistaken, as opposed to minimizing the probability of the discriminator being correct. In addition, the cost function was further modified by adding an L1 based regularizer. This will theoretically preserve the structure of the original images and prevent the generator from assigning arbitrary colors to pixels just to fool the discriminator:
<p align='center'>  
  <img src='img/gan_new.png' />
</p>

### Conditional GAN
In a traditional GAN, the input of the generator is randomly generated noise data z. However, this approach is not applicable to the automatic colorization problem due to the nature of its inputs. The generator must be modified to accept grayscale images as inputs rather than noise. This problem was addressed by using a variant of GAN called [conditional generative adversarial networks](https://arxiv.org/abs/1411.1784). Since no noise is introduced, the input of the generator is treated as zero noise with the grayscale input as a prior:
<p align='center'>  
  <img src='img/con_gan.png' />
</p>
The discriminator gets colored images from both generator and original data along with the grayscale input as the condition and tries to tell which pair contains the true colored image:
<p align='center'>  
  <img src='img/cgan.png' width='450px' height='368px' />
</p>

### Networks Architecture
The architecture of generator is inspired by  [U-Net](https://arxiv.org/abs/1505.04597):  The architecture of the model is symmetric, with `n` encoding units and `n` decoding units. The contracting path consists of 4x4 convolution layers with stride 2 for downsampling, each followed by batch normalization and Leaky-ReLU activation function with the slope of 0.2. The number of channels are doubled after each step. Each unit in the expansive path consists of a 4x4 transposed convolutional layer with stride 2 for upsampling, concatenation with the activation map of the mirroring layer in the contracting path, followed by batch normalization and ReLU activation function. The last layer of the network is a 1x1 convolution which is equivalent to cross-channel parametric pooling layer. We use `tanh` function for the last layer.
<p align='center'>  
  <img src='img/unet.png' width='700px' height='168px' />
</p>

For discriminator, we use similar architecture as the baselines contractive path: a series of 4x4 convolutional layers with stride 2 with the number of channels being doubled after each downsampling. All convolution layers are followed by batch normalization, leaky ReLU activation with slope 0.2. After the last layer, a convolution is applied to map to a 1 dimensional output, followed by a sigmoid function to return a probability value of the input being real or fake
<p align='center'>  
  <img src='img/discriminator.png' width='450px' height='168px' />
</p>
  
## Places365 Results
Colorization results with Places365. (a) Grayscale. (b) Original Image. (c) Colorized with GAN.
<p align='center'>  
  <img src='img/places365.jpg' />
</p>

## Authors

* **Yuxin Ding** 
* **Wen Xu** 

