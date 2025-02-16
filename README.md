# Deep Learning Homework 1 - IST 2023/24

This repository contains the solutions for Homework 1 of the Deep Learning course at Instituto Superior Técnico (IST). The assignment covers three topics: medical image classification using linear models and neural networks, implementation with an autodiff framework, and Boolean function computation with MLPs.

## Topics Covered
### 1. Medical Image Classification  
- Implementation of **Perceptron, Logistic Regression, and Multi-Layer Perceptron (MLP)** for medical image classification on the **OCTMNIST dataset**.  
- Comparison of **different learning rates** and training methods.  
- Training with **stochastic gradient descent** without using machine learning libraries.  
- Evaluation using accuracy metrics and visualization of training loss curves.  

### 2. Automatic Differentiation and Deep Learning Frameworks  
- Implementation of **logistic regression and neural networks using PyTorch**.  
- Comparison of training with different **batch sizes, learning rates, and dropout regularization**.  
- Performance analysis and hyperparameter tuning.  

### 3. Boolean Function Computation with MLPs  
- Implementation of a **multi-layer perceptron** to compute Boolean functions.  
- Analysis of **single-layer perceptrons vs. multi-layer perceptrons** in solving non-linearly separable problems.  
- Experiments with **hard-threshold activations and ReLU activations**.  

## Installation & Setup
To set up the environment and run the models, follow these steps:

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/deep-learning-hw1.git
   cd deep-learning-hw1
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Medical Image Classifier**  
   ```bash
   python hw1-q1.py
   ```

4. **Run the PyTorch-Based Models**  
   ```bash
   python hw1-q2.py
   ```

5. **Run the Boolean Function MLP**  
   ```bash
   python hw1-q3.py
   ```

## Results
- **Medical Image Classification:**  
  - Perceptron accuracy: **34.22%** (test set)  
  - Logistic Regression accuracy: **60.68%** (test set)  
  - MLP accuracy: **76.75%** (test set)  

- **Deep Learning Framework Models:**  
  - Best validation accuracy with PyTorch-based logistic regression: **64.17%**  
  - Best test accuracy with PyTorch-based MLP: **82.00%**  
  - Impact of batch size, dropout, and L2 regularization explored.  

- **Boolean Function Computation:**  
  - Demonstrated the **XOR problem cannot be solved with a single-layer perceptron**.  
  - Designed **MLP architectures** with different activation functions to solve Boolean functions.  

## License
This project is for educational purposes and follows an open-access policy.



