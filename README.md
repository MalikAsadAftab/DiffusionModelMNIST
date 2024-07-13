# Diffusion Probabilistic Model
This project implements a Conditional Diffusion Probabilistic Model (DDPM) for generating MNIST images. The implementation is based on the concepts described in the following papers:

- [DDPM: Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Classifier-free conditional DDPM](https://arxiv.org/abs/2207.12598)

## **What is Conditional DDPM?**
A conditional diffusion model is a modification of an unconditional diffusion model. An unconditional diffusion model treats the generation process as a Markov chain of steps that turn data (e.g. images) into noise and vice versa.

In the realm of diffusion models, noise injection is a crucial mechanism for generating realistic samples. This implementation focuses on implementing noise injection using the reparameterization trick within a diffusion model framework.

The implementation is based on MNIST dataset where it learns to generate MNIST digits, conditioned on a class label. The neural network architecture is a small U-Net (pretrained weights also available in this repo). This code is modified from this excellent repo which does unconditional generation. The diffusion model is a Denoising Diffusion Probabilistic Model (DDPM).

Project Structure
utils.py: Contains utility functions for data preprocessing and loading.
models.py: Entry point of the project that sets up and trains the model.
diffusion_model.py: Defines the conditional U-Net architecture and related components.
train.py: Contains the training loop and related functions.
testing.py: Functions for testing and evaluating the model.
requirements.txt: List of required libraries.


---

Feel free to reach out if you have any questions or need further assistance.

Happy Coding!

For further queries or help, don't hesitate to get in touch either via email: [asad.aftab@tuwien.ac.at](mailto:asad.aftab@tuwien.ac.at) or through [LinkedIn: Asad Aftab](https://www.linkedin.com/in/asad-aftab-malak/).

[![Email](https://img.icons8.com/color/48/000000/email.png)](mailto:asad.aftab@tuwien.ac.at)
[![LinkedIn](https://img.icons8.com/color/48/000000/linkedin.png)](https://www.linkedin.com/in/asad-aftab-malak/)
