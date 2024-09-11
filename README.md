# Densely Self-guided Wavelet Network (DSWN) for Image Denoising

This repository contains the implementation of a Densely Self-guided Wavelet Network (DSWN) for image denoising. The network combines wavelet decomposition with dense convolutional residual blocks to remove noise from images efficiently.

## Features
- **Wavelet Transform Integration**: Uses discrete wavelet transforms (DWT) to extract and denoise different frequency components of the image.
- **Dense Residual Blocks**: Implements dense convolutional blocks for feature extraction and improved denoising performance.
- **End-to-End Denoising**: The network outputs a denoised image by combining two branches: a residual branch and an end-to-end branch, which are averaged for final prediction.
  
## Code Overview
- `dwt2d(image)`: Performs 2D discrete wavelet transform on the input image.
- `idwt2d(coeffs)`: Performs inverse discrete wavelet transform to reconstruct the image.
- `dcr_block(x)`: Defines a convolutional residual block with PReLU activation.
- `dsw_network(input_shape)`: Builds the DSWN model, consisting of convolutional layers and wavelet-based processing.

## Running the Project
1. **Run the Main Script**:
   - To run the denoising model, execute the `main.py` file:
   ```bash
   python main.py
