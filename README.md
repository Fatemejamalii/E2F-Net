# E2F-Net: Eyes-to-Face Inpainting via StyleGAN Latent Space



## Description   
Official Implementation of the *E2F-Net* for both training and evaluation.

> **E2F-Net: Eyes-to-Face Inpainting via StyleGAN Latent Space**<br>
> Ahmad Hassanpour<sup></sup>, Fatemeh Jamalbafrani<sup></sup>, Bian Yang<sup></sup>, Kiran Raja<sup></sup>, Raymond Veldhuis<sup></sup>, Julian Fierrez<sup></sup><br>
> <p align="justify"><b>Abstract:</b> <i>Face inpainting refers to the task of filling the missing or deteriorated regions in face images with approximate content close to ground truth. Inpainting has been used in some applications such as face recognition under occlusions, and in general any image/video analysis application on low quality, uncontrolled, or in-the-wild acquisition conditions. In face inpainting, the approximate areas not only should look realistic but also approximate/generate an individual's identity and non-identity features. The complexity of task which is high given the wide range of demographics of faces, such as gender, ethnicity, skin-color, pose, and expression. The aim of this paper is to inpaint a face given periocular region (eyes-toface). A new Generative Adversarial Network (GAN)-based model called Eyes-to-Face Network (E2F-Net) is proposed in this work to achieve this. To extract identity and non-identity features from the periocular region, two encoders have been used. Then the extracted features are mapped to the latent space of a pre-trained StyleGAN generator leveraging both its state-of-the-art performance and its rich and expressive latent space without an additional burden of training it. Finally, to further improve the StyleGAN’s output, an optimization for GAN inversion technique has been proposed to accurately find the optimal code in the latent space. Further, our E2F-Net requires a minimum training process reducing the computational complexity as illustrated later. Through extensive experiments, we show that our method successfully reconstructs the whole face with high quality, surpassing current techniques, despite significantly less training and supervision efforts. We have generated seven eyes-to-face datasets based on wellknown public face datasets for training and verifying our proposed methods.</i></p>

## Setup

To setup everything you need check out the requirements.

## Training

### Preparing the Dataset
