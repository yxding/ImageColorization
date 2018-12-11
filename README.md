# Image Colorization with GANs 
In this project, we tried to approach the image colorization problem by using a conditional generative adversarial network (CGAN) and a Wasserstein generative adversarial network (WGAN). The networks were trained on two public datasets: [Stanford dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) and [VGG flower dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). The colorized results of CGAN and WGAN are [shown here.](#places365-results) For the result evaluation of CGAN, We achieved a 42% perceptual realism in the [turing test.] 

## Prerequisites
- Python 3.6
- Google Computer Engine with NVIDIA GPU (V100, 200G memory)
- Linux
- Tensorflow 1.7
- CUDA 
- CUDNN

## Getting Started
### Installation
Clone this repo:
```bash
https://github.com/yxding/ImageColorization.git
```

### Dataset
- [Stanford dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [VGG flower dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).
  
  After downloading the data, put them under the `datasets` folder.

### Training
- To train the model on dogs dataset with tuned hyperparameters using CGAN:
```
cd Colorizing-with-CGANs
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

- To train the model on flowers dataset with tuned hyperparameters:
```
cd Colorizing-with-CGANs
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

- To train the model on datasets with WGAN, just replace
```
cd Colorizing-with-CGANs
```
to
```
cd Colorizing-with-WGANs
```
and execute what has been done above

### Evaluate and Sample
- To evaluate the model and sample colorized outputs on the test-set, we first plot the loss and accuracy per epoch and choose the checkpoint which records the best model in the `checkpoints` folder. Then copy that checkpoint to the `test` folder in `checkpoints`.
- Run `test-eval.py` script:
```bash
python test-eval.py
```

## Approach

### Conditional GAN
The network architecture is shown as follows:
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


## Results Comparison between CGAN and WGAN
Figure below shows several colorized results of CGAN and WGAN. It is illustrated that WGAN can improve the results in some
cases, although the improvement is not very significant.
<p align='center'>  
  <img src='Images/wgan_compare.png' />
</p>

## Results of Perceptual Realism
Based on the feedback from 50 participants, it turns out that we can fool them by about 42%! Here are some examples from our survey and the number on the left of each picture shows the percentage of participants that recognized colorized image as the true one. The far left column shows example pairs that fool the participants most.
<p align='center'>  
  <img src='Images/compare_picture.png' />
</p>
## Authors

* **Yuxin Ding** 
* **Wen Xu** 

