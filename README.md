# E2F-Net: Eyes-to-Face Inpainting via StyleGAN Latent Space



## Description   
Official Implementation of the *E2F-Net* for both training and evaluation.

> **E2F-Net: Eyes-to-Face Inpainting via StyleGAN Latent Space**<br>
> Ahmad Hassanpour<sup></sup>, Fatemeh Jamalbafrani<sup></sup>, Bian Yang<sup></sup>, Kiran Raja<sup></sup>, Raymond Veldhuis<sup></sup>, Julian Fierrez<sup></sup><br>
> <p align="justify"><b>Abstract:</b> <i>Face inpainting, the technique of restoring missing or damaged regions in facial images, is pivotal for applications like face recognition in occluded scenarios and image analysis with poor-quality captures. This process not only needs to produce realistic visuals but also preserve individual identity characteristics. The aim of this paper is to inpaint a face given periocular region (eyes-to-face) through a proposed new Generative Adversarial Network (GAN)-based model called Eyes-to-Face Network (E2F-Net). The proposed approach extracts identity and non-identity features from the periocular region using two dedicated encoders have been used. The extracted features are then mapped to the latent space of a pre-trained StyleGAN generator to benefit from its state-of-the-art performance and its rich, diverse and expressive latent space without any additional training. We further improve the StyleGAN’s output to find the optimal code in the latent space using a new optimization for GAN inversion technique. Our E2F-Net requires a minimum training process reducing the computational complexity as a secondary benefit. Through extensive experiments, we show that our method successfully reconstructs the whole face with high quality, surpassing current techniques, despite significantly less training and supervision efforts. We have generated seven eyes-to-face datasets based on well-known public face datasets for training and verifying our proposed methods.</i></p>

## Setup

To setup everything you need check out the requirements.

## Training

### Preparing the Dataset
