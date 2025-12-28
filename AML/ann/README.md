# Artificial Neural Network (ANN) from Scratch
A mathematical implementation of a Multi-Layer Perceptron (MLP) to predict the impact of gaming habits on academic performance.

## Table of Contents
1. Project Overview
2. Dataset Description
3. Mathematical Foundation
4. Installation & Usage
5. Project Structure
6. Results

## Project Overview
This project implements a neural network entirely from scratch using only Python and NumPy. It bypasses high-level frameworks like TensorFlow or PyTorch to demonstrate the underlying calculus and linear algebra that power deep learning.

The model is trained on the Gaming Hours vs. Academic Performance dataset to perform binary classification. It determines whether a student's gaming habits are having a "Negative" impact on their performance based on sleep, stress, and study scores.

## Key Features
1. Vectorized Operations: efficient matrix multiplications using numpy.dot.
2. Custom Backpropagation: manual implementation of the Chain Rule for gradient descent.
3. Dynamic Architecture: scalable NeuralNetwork class supporting variable layer sizes.
4. Data Pipeline: automated normalization (Min-Max scaling) and train/test splitting.

## Dataset Description
The model utilizes Gaming_Hours_vs_Academic_Performance.csv.

    Feature              Type        Description
    Daily_Gaming_Hours   Float       Average hours spent gaming per day.
    Sleep_Hours          Float       Average sleep duration.
    Stress_Level         Int         Self-reported stress scale (1-10).
    Academic_Score       Int         Current performance score (0-100).
    Performance_Impact   Target      The label to predict (Negative vs. Neutral/Positive).

Note: The target variable is binarized: 1 for Negative Impact, 0 for others.

## Mathematical Foundation
This implementation relies on the following core equations:

### 1. Forward Propagation

For a single layer, the output $Z$ and activation $A$ are calculated as:

$$Z = W \cdot X + b$$
$$A = \sigma(Z) = \frac{1}{1 + e^{-Z}}$$
(Where $\sigma$ is the Sigmoid activation function)

### 2. Loss Function (MSE)

We minimize the Mean Squared Error to measure performance:

$$Loss = \frac{1}{n} \sum (y_{true} - y_{pred})^2$$

### 3. Backpropagation (Chain Rule)

Weights are updated by calculating the gradient of the Loss with respect to Weights ($W$):

$$\frac{\partial Loss}{\partial W} = \frac{\partial Loss}{\partial A} \cdot \frac{\partial A}{\partial Z} \cdot \frac{\partial Z}{\partial W}$$

## Installation & Usage
### Prerequisites
Ensure you have Python installed along with the required libraries:

    Bash
    pip install numpy pandas matplotlib

### Running the Model
1. Clone the repository.
2. Ensure Gaming_Hours_vs_Academic_Performance.csv is in the root directory.
3. Run the notebook.

## Project Structure

    ├── Gaming_Hours_vs_Academic_Performance.csv    # Dataset
    ├── ann.ipynb                                   # Main implementation notebook
    ├── README.md                                   # Documentation

## Results
After training for 10,000 epochs, the model achieves convergence.
1. Final Loss: $< 0.04$
2. Test Set Accuracy: $~98\%$

## Sample Prediction Output

    Epoch 10000 | Loss: 0.03412
    --------------------------------------------------
    Student A (High Gaming, Low Sleep):
        > Predicted: NEGATIVE IMPACT (Prob: 0.98)
        > Actual:    NEGATIVE IMPACT

    Student B (Low Gaming, High Sleep):
        > Predicted: Neutral/Positive (Prob: 0.12)
        > Actual:    Neutral/Positive
