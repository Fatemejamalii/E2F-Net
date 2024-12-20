# E2F-Net: Eyes-to-Face Inpainting via StyleGAN Latent Space



## Description   
Official Implementation of the *E2F-Net* for both training and evaluation.

> **E2F-Net: Eyes-to-Face Inpainting via StyleGAN Latent Space**<br>
> Ahmad Hassanpour<sup></sup>, Fatemeh Jamalbafrani<sup></sup>, Bian Yang<sup></sup>, Kiran Raja<sup></sup>, Raymond Veldhuis<sup></sup>, Julian Fierrez<sup></sup><br>
> <p align="justify"><b>Abstract:</b> <i>Face inpainting, the technique of restoring missing or damaged regions in facial images, is pivotal for applications like face recognition in occluded scenarios and image analysis with poor-quality captures. This process not only needs to produce realistic visuals but also preserve individual identity characteristics. The aim of this paper is to inpaint a face given periocular region (eyes-to-face) through a proposed new Generative Adversarial Network (GAN)-based model called Eyes-to-Face Network (E2F-Net). The proposed approach extracts identity and non-identity features from the periocular region using two dedicated encoders have been used. The extracted features are then mapped to the latent space of a pre-trained StyleGAN generator to benefit from its state-of-the-art performance and its rich, diverse and expressive latent space without any additional training. We further improve the StyleGAN’s output to find the optimal code in the latent space using a new optimization for GAN inversion technique. Our E2F-Net requires a minimum training process reducing the computational complexity as a secondary benefit. Through extensive experiments, we show that our method successfully reconstructs the whole face with high quality, surpassing current techniques, despite significantly less training and supervision efforts. We have generated seven eyes-to-face datasets based on well-known public face datasets for training and verifying our proposed methods.</i></p>

<p align="center">
<img src="E2F-NET_fig.jpg"/>
</p>

## Setup

To setup everything you need check out the requirements.


## Installation

Clone this repo:
```
git clone https://github.com/fatemejamalii/E2F-Net && cd E2F-Net
```

## Dataset
We conduct all experiments on our generated dataset called [E2F-Net_db](https://drive.google.com/drive/folders/1CPsk5wyO8VW5JoabWOIN39Bt9RStcgqB?usp=drive_link) extracted from the well-known [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) dataset. To extract the periocular region from each face image, the images are reshaped to size  256 ×256 and then by utilizing a landmark detector , eyes are detected. Doing this, I_m and I_c are produced for each image. Moreover, we removed misleading samples including those eyes covered by sunglasses or faces that have more than 45 degrees in one angle (roll, pitch, yaw) leading to hiding one of the eyes by using WHENet algorithms. Finally, the total number of 
samples is 24,554 among which 22,879 will be used for the training process and the rest, which is 1,685 images, for the test.

# Getting Started
To use the pretrained models and checkpoints, download them from the following links then copy them to the pretrained folder and the checkpoints folder.

[Weights](https://drive.google.com/drive/folders/18hGuGPvW3cHdqDt8yVCJsNjOeDDpO2pG?usp=drive_link)

### Quick Testing
```
python test.py
NAME
--pretrained_models_path ./pretrained/arcface_weights/
--load_checkpoint ./checkpoints
--arcface_checkpoints ./pretrained/arcface_checkpoint/
--id_dir VAL_IMAGES
--mask_dir VAL_IMAGES
--eye_dir VAL_EYES
--eye_dataset VAL_EYES
--output_dir OUTPUT_PATH
--test_func opt_infer_pairs
--epochnum EPOCH_NUM
```

### Training
```
python main.py 
NAME
--resolution 256
--pretrained_models_path ./pretrained
--load_checkpoint ./checkpoints
--arcface_checkpoints ./pretrained/arcface_checkpoint/
--results_dir RESULTS_DIR
--dataset_path DATASET256
--celeba_path CELEBA_DATASET_PATH 
--celeba_ws_path CELEBA_DATASET_PATH
--eye_dataset CELEBA_DATASET_PATH/eyes
--batch_size BATCHSIZE
--cross_frequency 0
--train_data_size 24554
--test_frequency 1000 
--no_train_real_attr 
--no_test_real_attr
--no_test_with_arcface
--wich_dataset celeba
--initial_epoch 0 
--no_W_D_loss
```
