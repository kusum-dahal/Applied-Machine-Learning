# CNN for Skin Lesion Classification from scratch
A low-level implementation of a Convolutional Neural Network (CNN) built exclusively using NumPy and Pandas to demonstrate the explicit mathematical mechanics of computer vision, including manual implementations of forward propagation, gradient calculation, and parameter optimization.

## Table of Contents
1. Project Overview
2. Dataset Description
3. Mathematical Foundation
4. Installation & Usage
5. Project Structure
6. Results

## Project Overview
The architecture implements a standard computer vision pipeline—Convolution, Pooling, and Fully Connected classification—constructed from first principles within a Jupyter Notebook environment.

## Key Technical Implementations:
1. Vectorized Forward Pass: Efficient sliding-window operations using NumPy array manipulation.
2. Manual Backpropagation: Derivation and implementation of gradients for convolutional filters ($dL/df$) and dense weights ($dL/dW$).
3. Stochastic Gradient Descent (SGD): Weight updates performed purely via matrix operations.
4. Data Handling: Utilization of Pandas for tracking training metrics and history.
5. Zero-Framework Architecture: The model relies solely on standard arithmetic libraries, ensuring transparency in the learning algorithm.

## Dataset Description
The model operates on a synthetic dataset designed to simulate binary classification tasks in medical imaging (Benign vs. Malignant).
1. Dimensions: $28 \times 28 \times 1$ (Grayscale).
2. Sample Size: 500 Training samples, 20 Test samples.
3. Class Logic:
    1. Class 0 (Benign): Uniform noise distribution with high luminance base.
    2. Class 1 (Malignant): Low luminance regions with irregular geometric masking and Gaussian noise artifacts.

Note: The data generation pipeline is embedded directly in the source code to ensure reproducibility without external file dependencies.

## Mathematical Foundation
The implementation relies on the chain rule to propagate error through the network layers.

### Convolutional Gradients
For a filter $F$, the gradient is computed by convolving the input slice $X$ with the upstream gradient $\frac{\partial L}{\partial Y}$:
$$\frac{\partial L}{\partial F_{m,n}} = \sum_{i,j} X_{i+m, j+n} \cdot \frac{\partial L}{\partial Y_{i,j}}$$

### Max Pooling Backpropagation
Max pooling is non-differentiable in the traditional sense. The gradient is sparse and is routed exclusively to the index of the maximum value within the receptive field:
$$\frac{\partial L}{\partial x_{i,j}} = \begin{cases} \frac{\partial L}{\partial y} & \text{if } x_{i,j} = \max(window) \\ 0 & \text{otherwise} \end{cases}$$

###Optimization (Cross-Entropy & Softmax)
The network minimizes Categorical Cross-Entropy. For the final layer, the gradient of the loss $L$ with respect to the logits $z$ simplifies to the difference between the prediction vector $\hat{y}$ and the one-hot encoded label $y$:
$$\frac{\partial L}{\partial z} = \hat{y} - y$$

## Installation & Usage

### Dependencies
The environment requires only standard scientific computing libraries.

    Bash
    pip install numpy pandas matplotlib ipykernel

### Execution (VS Code)
This project is designed to run as a Jupyter Notebook within Visual Studio Code.
1. Clone this repository.
2. Open the project folder in VS Code.
3. Open the Notebook file.
4. Select Kernel: Ensure your Python environment with the installed dependencies is selected as the kernel.
5. Run All: Execute the cells sequentially.
6. Data Generation: Creates the synthetic dataset.
7. Class Definitions: Initializes the custom Conv3x3, MaxPool2, and Softmax layers.
8. Training Loop: Runs the SGD optimization (approx. <2 mins on CPU).
9. Evaluation: Plots loss curves and test sample predictions.

## Project Structure
The codebase is modularized into class-based definitions for each network component.

    Component               Description
    Conv3x3                 Implements 3 X 3 convolution with Xavier Initialization. Handles 3D volume preservation.
    MaxPool2                Performs 3 X 3 downsampling. Caches indices of max values for accurate backward pass routing.
    Softmax                 Dense layer transforming flattened feature maps into probability distributions.
    train()                 Orchestrates the forward/backward passes.
    Pandas Integration      pd.DataFrame is used to structure and visualize the loss/accuracy history over epochs.

## Results
Upon training for 3 epochs, the model demonstrates rapid convergence due to the distinct feature separation in the synthetic dataset.
1. Performance: >90% Accuracy on the held-out test set.
2. Visualization: The notebook generates a Matplotlib figure showing:
    1. Training Metrics: Loss vs. Accuracy trends over time.
    2. Inference: A sample grid of input images, Ground Truth labels, Predicted labels, and Confidence scores.