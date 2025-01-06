Implemented [Distraction is All You Need: Memory-Efficient Image Immunization against Diffusion-Based Image Editing](https://openaccess.thecvf.com/content/CVPR2024/papers/Lo_Distraction_is_All_You_Need_Memory-Efficient_Image_Immunization_against_Diffusion-Based_CVPR_2024_paper.pdf) paper from scratch.

## The mechanism for immunization is based on attacking the cross-attention layers of a denoising U-Net.
- Creating mask by averaging cross-attention maps correspondent to a token
- Token represents immunized object
- Applying mask on the image
- 2 cycles: epochs and diffusion
- Calculating loss using L1 norm of the averaged attention responses for different diffusion steps
- Estimating perturbations using the projected gradient descent on the immunized image
- Applying the estimated perturbations on the image for each attacking step

## Disussions&Conclusions:

- The absence of code and details of implementation in the paper make it hard to reproduce
- The model is quite slow: it takes 20-30 minutes on 1 image using A100 GPU on Colab
- It takes 15Gb of memory instead of 12Gb proposed in the paper, field for optimization (or different implementation)
- Not zeroing deltas on each diffusion step and clipping them doesn’t seem intuitive and robust
- In further experimentation, try to introduce loss for the perturbation
