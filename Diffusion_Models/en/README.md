# Generative AI with Diffusion Models - Notes

### Part 1: From U-Nets to Diffusion
Part 1 discusses the early history of image generation AI and introduces a key concept: transposed convolution. This technique enables GANs for generating new images and U-Nets in segmentation networks. The differences between transposed convolution and deconvolution are also compared.
- **Key Topics**: U-Nets, GANs, Image Segmentation, Transposed Convolution.

### Part 2: Denoising Diffusion Probabilistic Models
Part 2 describes the Denoising Diffusion Probabilistic Model (DDPM). The principle of diffusion is likened to dropping ink into clear water; over time, the entire glass turns gray. Image generation uses a reverse diffusion process to create pictures.
- **Key Topics**: Forward and reverse diffusion, ELBO, model architecture for denoising.

### Part 3: Optimizations
Part 3 lists optimization methods for model architecture, including visualization dashboard optimization, Group Normalization (GN), GELU, and rearrange pooling. Visualization dashboard optimization refers to making generated images more recognizable. GN helps models converge faster, with other methods like BN, LN, and IN also mentioned. GELU is an activation function similar to ReLU, with the advantage that gradients less than zero do not become zero. Rearrange pooling, unlike max pooling, allows the model to decide which features to retain.
- **Key Topics**: Checkerboard problem, Group Normalization, Sinusoidal Position Embeddings, Rearrange pooling.

### Part 4: Classifier-Free Diffusion Guidance
Part 4 introduces Classifier Free Guidance, one of the current mainstream ideas in diffusion model development. The advantage of this method is that when training a diffusion model, there’s no need to train a corresponding classifier for each category.
- **Key Topics**: Bernoulli masks, weighted reverse diffusion.

### Part 5: CLIP
Part 5 discusses Contrastive Language–Image Pre-training (CLIP), a method that trains models by matching textual descriptions with images, enabling the model to generate corresponding textual descriptions for unseen images.
- **Key Topics**: Matching text to image using cosine similarity, training dynamics in CLIP.

### Part 6: State of the Art Models
Part 6 provides an overview of various model architectures such as VAE, GAN, Diffusion, and CLIP, comparing their differences and expressing the hope for building Trustworthy AI in the future.
- **Key Topics**: Latent Diffusion, Variational Autoencoders (VAE), Trustworthy AI.


![DLI Header](images/DLI_Header.png)


Next Steps

- Check out [NVIDIA's Deep Learning Blog](https://blogs.nvidia.com/blog/category/deep-learning/) to see the concepts learned in class used in action.
- Explore the [NGC AI Playground](https://catalog.ngc.nvidia.com/playground) with interactive apps demoing Generative AI.
- Learn how to [deploy models at scale](https://courses.nvidia.com/courses/course-v1:DLI+S-FX-03+V1/).
- Setup your own JupyterLab environment using the [container for this course](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch):

```
FROM nvcr.io/nvidia/pytorch:23.05-py3

RUN pip install --upgrade pip
RUN pip install torchview einops==0.6.1 graphviz==0.20.1 ipywidgets==8.1.0
RUN pip install git+https://github.com/openai/CLIP.git

RUN apt-get -y update
RUN apt-get -y install graphviz
```

Check out [Generative AI by Getty Images](https://www.gettyimages.com/ai/generation/about), powered by NVIDIA.