# ShapeAtlas-UNet: Enhancing U-Net Performance for Liver Segmentation


## Read about this project in detail from this [Paper](https://github.com/yubrajbhandari923/shapeatlasUnet/blob/master/Improving_Unets.pdf)

## Project Overview

ShapeAtlas-UNet is an innovative approach to improve the performance of U-Net models for liver segmentation in CT images, particularly focusing on enhancing efficiency for smaller networks and resource-constrained environments. This project introduces a method that incorporates shape atlas information into the U-Net architecture, aiming to boost segmentation accuracy without significantly increasing computational demands during inference.

## Key Features

- **Dual Encoder Architecture**: Incorporates a main encoder for CT images and a separate encoder for shape atlas information.
- **Shape Atlas Integration**: Utilizes a pre-computed probabilistic shape atlas to provide prior knowledge about liver shape and location.
- **Efficient Inference**: Pre-computes the shape atlas latent representation to reduce computational overhead during inference.
- **Performance Boost for Small Models**: Significantly improves segmentation accuracy for models with limited latent space dimensions.

## Methodology

1. **Shape Atlas Generation**: Creates a probabilistic shape atlas by averaging aligned liver segmentation masks from the training dataset.
2. **Enhanced U-Net Architecture**: Implements a dual-encoder U-Net with parallel pathways for CT image and shape atlas processing.
3. **Latent Space Fusion**: Concatenates the latent representations from both encoders before decoding.
4. **Optimized Inference**: Uses pre-computed shape atlas latent representation during inference to maintain efficiency.

## Results

Experiments on the Abdomen Atlas mini dataset demonstrated:
- Substantial improvements for smaller models (e.g., 8% Dice score increase for 32-dimensional latent space)
- Enhanced data efficiency, achieving better performance with fewer training examples
- Minimal increase in parameters and computational requirements

## Potential Applications

This approach is particularly beneficial for:
- Deployment on low-end devices or in resource-constrained environments
- Scenarios with limited training data availability
- Improving accessibility of advanced medical image segmentation techniques in diverse healthcare settings

## Future Directions

- Extending the approach to multi-organ segmentation
- Investigating transfer learning capabilities
- Exploring integration with other segmentation architectures
