# RNN for Word Prediction from scratch
The project demonstrates the scratch implementation of a Recurrent Neural Network (RNN) as a fundamental mechanics of Natural Language Processing (NLP) by building a character/word-level predictor without the use of high-level deep learning frameworks (e.g., PyTorch, TensorFlow). It manually implements Backpropagation Through Time (BPTT), hidden state management, and temperature-scaled text generation.

## Table of Contents
1. Project Overview
2. Dataset Description
3. Mathematical Foundation
4. Installation & Usage
5. Project Structure
6. Results

## Project Overview
The objective of this project is to demystify sequential learning algorithms. By implementing an RNN at the matrix level, this notebook exposes the internal "memory" mechanism that allows neural networks to process time-series data.

### Key Technical Implementations:
1. Hidden State Recurrence: Manual implementation of the feedback loop ($h_t = f(h_{t-1}, x_t)$).
2. Backpropagation Through Time (BPTT): Unrolling the network to calculate gradients across time steps.
3. Gradient Clipping: Implementation of gradient scaling to prevent the "Exploding Gradient" problem common in vanilla RNNs.
4. Temperature Sampling: A probability scaling mechanism to control the randomness/creativity of the generated text.

## Dataset Description
The model is trained on a synthetic micro-corpus designed to test specific linguistic patterns (Contextual Association).
Source: Embedded synthetic text strings (e.g., "Python is used for data science", "The quick brown fox...").
Preprocessing:
Tokenization: Word-level splitting with punctuation removal.
Encoding: One-Hot Encoding for input vectors.Vocabulary: Dynamically generated based on the input corpus size.

##  Mathematical Foundation
The core of the RNN is the hidden state update, which serves as the network's short-term memory.

### Forward Pass (Recurrence)
At every time step $t$, the new hidden state $h_t$ is calculated using the current input $x_t$ and the previous hidden state $h_{t-1}$:

$$h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$$

The output (logits) $y_t$ is then computed from this hidden state:

$$y_t = W_{hy} \cdot h_t + b_y$$

### Probability Distribution (Softmax)
To predict the next word, we apply the Softmax function with temperature scaling ($T$):

$$P(y_t) = \frac{e^{y_t / T}}{\sum e^{y_j / T}}$$

###  Loss Function
The network minimizes the Cross-Entropy Loss between the predicted probability distribution and the actual next word:

$$L = - \sum y_{true} \cdot \ln(y_{pred})$$

## Installation & Usage

### Dependencies
The project requires only standard scientific computing libraries.

    Bash
    pip install numpy pandas matplotlib

### Execution
1. Clone this repository.
2. Open the Jupyter Notebook in VS Code or Jupyter Lab.
3. Run All Cells:
    1. Data Preparation: Tokenizes the text and creates training pairs.
    2. Model Definition: Initializes the VanillaRNN class with random weights.
    3. Training: optimizing weights via Stochastic Gradient Descent (SGD).
    4. Evaluation: visualize loss curves and generate sample text.

## Project Structure
The codebase is organized into modular classes:

    Component           Description
    TextProcessor       Handles tokenization, vocabulary building, and One-Hot mapping (Word <-> Index).
    VanillaRNN          The core neural network class. 
                        • forward(): Computes the sequence pass.
                        • backward(): Computes BPTT and gradients.
                        • predict(): Generates new text sequences.
    Training Loop       Manages epochs, calculates accuracy, and logs metrics to a Pandas DataFrame.
    Visualization       Uses Matplotlib to plot Loss and Accuracy trends over time.

## Results
The model typically converges within 1000-2000 epochs on the micro-corpus.
1. Convergence: The loss curve shows a standard logarithmic decay.
2. Text Generation:
    1. Strict Sampling ($T=0.5$): The model faithfully reproduces the training sentences (e.g., "The quick brown fox jumps...").
    2. Creative Sampling ($T=1.0+$): The model attempts to mix contexts, occasionally producing grammatically correct but novel combinations.

## Limitations
1. Vanishing Gradients: As a "Vanilla" RNN, the model struggles with long-term dependencies (sequences longer than 10-15 words)
2. Memory: The hidden_size is kept small for computational efficiency in NumPy.