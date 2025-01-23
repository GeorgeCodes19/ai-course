# Chapter 1: Math & ML Fundamentals

> **Note**: This chapter lays the groundwork for the entire textbook. It covers essential mathematical concepts (Linear Algebra, Probability/Statistics, and Calculus) and introduces fundamental Machine Learning (ML) models (Regression and Classification). Mastery of these topics is crucial for understanding how modern AI—and particularly Large Language Models—operate under the hood.

---

## Table of Contents

1. [Chapter Overview](#chapter-overview)  
2. [Linear Algebra for Machine Learning](#linear-algebra-for-machine-learning)  
   1. [Vectors and Matrices](#vectors-and-matrices)  
   2. [Eigenvalues, Eigenvectors, and SVD](#eigenvalues-eigenvectors-and-svd)  
   3. [Matrix Operations in ML Frameworks](#matrix-operations-in-ml-frameworks)  
   4. [Practice Exercises (Linear Algebra)](#practice-exercises-linear-algebra)  
3. [Probability, Statistics, and Calculus](#probability-statistics-and-calculus)  
   1. [Basic Probability and Distributions](#basic-probability-and-distributions)  
   2. [Statistics for Model Evaluation](#statistics-for-model-evaluation)  
   3. [Calculus and Optimization](#calculus-and-optimization)  
   4. [Practice Exercises (Prob/Stats/Calc)](#practice-exercises-probstatscalc)  
4. [Basic ML Models: Regression and Classification](#basic-ml-models-regression-and-classification)  
   1. [Linear Regression](#linear-regression)  
   2. [Logistic Regression](#logistic-regression)  
   3. [Overfitting and Regularization](#overfitting-and-regularization)  
   4. [Practice Exercises (Basic ML)](#practice-exercises-basic-ml)  
5. [Summary and Further Reading](#summary-and-further-reading)  
6. [Chapter 1 Dictionary](#chapter-1-dictionary)

---

## Chapter Overview

Modern neural networks and especially **Large Language Models (LLMs)** rely on solid mathematical foundations. Before diving into deep learning specifics, it’s critical to understand:

- **Linear Algebra**: The language of vector and matrix operations, which form the building blocks of neural network computations.  
- **Probability & Statistics**: Fundamental for interpreting model outputs and dealing with uncertainties.  
- **Calculus**: Powers the optimization processes (like gradient descent) used to train models.  
- **Basic Machine Learning Techniques**: Simple regression and classification approaches help illustrate how more complex neural models are built.

By the end of this chapter, you should be able to:

1. Perform basic matrix/vector operations and understand how they translate into ML frameworks.  
2. Describe core probability distributions and statistical measures used in model evaluation.  
3. Explain derivatives, partial derivatives, and the chain rule in the context of gradient-based optimization.  
4. Implement and evaluate simple ML models (linear regression, logistic regression) and discuss overfitting/regularization.

> **Real-World Example**  
> - **House Price Prediction**: Linear regression can be used to forecast house prices based on features like square footage, location, and number of bedrooms.  
> - **Email Spam Detection**: Logistic regression can classify emails as spam or not spam using word frequencies.  
> - **LLM Foundations**: All these fundamental math operations (matrix multiplication, gradient descent) will later be applied to more sophisticated architectures like Transformers.

---

## Linear Algebra for Machine Learning

### Vectors and Matrices

**Definition & Notation**

- A **vector** is a one-dimensional array (or list) of numbers.  
  - Example in 2D space: \(\mathbf{v} = (2, 5)\).  
  - Example in higher dimensions: \(\mathbf{x} = (x_1, x_2, \dots, x_d)\).

- A **matrix** is a two-dimensional array of numbers arranged in rows and columns.  
  - Example:  
    \[
      A = \begin{pmatrix}
      1 & 2 & 3 \\
      4 & 5 & 6
      \end{pmatrix}
    \]

**Common Operations**

1. **Matrix Addition**  
   If \(A\) and \(B\) have the same shape,  
   \[
     C = A + B, \quad \text{where } C_{ij} = A_{ij} + B_{ij}.
   \]

2. **Matrix-Vector Multiplication**  
   \[
     \mathbf{y} = A \mathbf{x}, \quad A \in \mathbb{R}^{m\times n},\ \mathbf{x}\in \mathbb{R}^{n},\ \mathbf{y}\in \mathbb{R}^{m}.
   \]
   The result \(\mathbf{y}\) is an \(m\)-dimensional vector.

3. **Matrix-Matrix Multiplication**  
   \[
     C = A B,\quad C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}.
   \]
   The order of multiplication matters (i.e., \(A B \neq B A\) in general).

4. **Transpose**  
   \[
     A^T_{ij} = A_{ji}.
   \]
   Flips a matrix over its diagonal.

**Why It Matters in ML**

- **Feature Representation**: A dataset of \(N\) samples with \(d\) features is often stored as a matrix \(X \in \mathbb{R}^{N \times d}\).  
- **Neural Network Weights**: Each layer typically has a weight matrix \(W\) and a bias vector \(\mathbf{b}\).

**Step-by-Step Example**

Suppose you have a small dataset \(X\) with 2 samples (\(N=2\)) and 3 features (\(d=3\)):

\[
  X = \begin{pmatrix}
  x_{11} & x_{12} & x_{13} \\
  x_{21} & x_{22} & x_{23}
  \end{pmatrix}
\]
And a weight matrix \(W \in \mathbb{R}^{3 \times 2}\) that transforms a 3D input into a 2D output:

\[
  W = \begin{pmatrix}
  w_{11} & w_{12} \\
  w_{21} & w_{22} \\
  w_{31} & w_{32}
  \end{pmatrix}
\]

1. **Matrix multiplication** \(X W\) yields an \((2 \times 2)\) matrix:
   \[
     X W = \begin{pmatrix}
     x_{11} & x_{12} & x_{13} \\
     x_{21} & x_{22} & x_{23}
     \end{pmatrix}
     \begin{pmatrix}
     w_{11} & w_{12} \\
     w_{21} & w_{22} \\
     w_{31} & w_{32}
     \end{pmatrix}
   \]

2. **Output interpretation**:
   Each row of the result corresponds to a sample, and each column corresponds to a transformed feature dimension.

---

### Eigenvalues, Eigenvectors, and SVD

**Eigenvalues and Eigenvectors**

For a square matrix \(A \in \mathbb{R}^{n \times n}\), an **eigenvector** \(\mathbf{v}\) and its **eigenvalue** \(\lambda\) satisfy:

\[
  A \mathbf{v} = \lambda \mathbf{v}.
\]

- Geometric interpretation: \(\mathbf{v}\) is scaled by \(\lambda\) but not rotated by \(A\).  
- **Real-World Use Case**: In finance, covariance matrices can be diagonalized using eigenvalues/eigenvectors, revealing the principal directions of market movement (PCA).

**Singular Value Decomposition (SVD)**

Every matrix \(M \in \mathbb{R}^{m \times n}\) can be factored into:

\[
  M = U \Sigma V^T,
\]
where:

- \(U \in \mathbb{R}^{m \times m}\) is orthonormal,  
- \(\Sigma \in \mathbb{R}^{m \times n}\) is diagonal (with non-negative real numbers on the diagonal),  
- \(V \in \mathbb{R}^{n \times n}\) is orthonormal.

**Why Important?**

- **PCA**: You can perform dimensionality reduction by using the largest singular values.  
- **Noise Filtering**: SVD can identify low-rank approximations.

---

### Matrix Operations in ML Frameworks

Modern ML libraries—such as [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), and [JAX](https://github.com/google/jax)—optimize these operations with parallelization:

- **Batch Multiplication**: In deep networks, you multiply matrices for multiple samples in a single pass.  
- **Automatic Differentiation**: The framework constructs a computational graph for operations, then automatically computes derivatives (gradients).

**Example (PyTorch)**

```python
import torch

# Creating a random matrix A (3x3) and vector x (3x1)
A = torch.randn(3, 3)
x = torch.randn(3, 1)

# Matrix-vector multiplication
y = A @ x  # or torch.matmul(A, x)
print("Result vector (y):", y)

Practice Exercises (Linear Algebra)
	1.	Matrix-Vector Multiplication
	•	Create a (3\times 3) matrix and a 3D vector. Multiply them. Verify the result manually.
	2.	Eigen-Decomposition
	•	Take a (2\times 2) matrix, find its eigenvalues and eigenvectors (manually or using NumPy). Interpret the meaning in terms of data transformation.
	3.	SVD on an Image
	•	Load a small grayscale image as a matrix. Compute SVD. Try reconstructing the image using only the top 20% of singular values. Observe the compression effect.

Probability, Statistics, and Calculus

Basic Probability and Distributions

Core Concepts
	•	Random Variable: A variable whose possible values are numerical outcomes of a random process.
	•	Probability Distribution: Describes how likely each outcome is. Examples:
	•	Bernoulli((p)): One trial, returns 1 with probability (p), else 0.
	•	Binomial((n, p)): Sum of (n) Bernoulli trials.
	•	Normal (Gaussian)((\mu, \sigma^2)): Continuous distribution with mean (\mu) and variance (\sigma^2).

	Step-by-Step Example
		•	Bernoulli((p=0.3)) means there’s a 30% chance of “success” (1), 70% chance of “failure” (0).
	•	Real-world: Email open rates. Suppose any given email has a 30% chance to be opened.

Statistics for Model Evaluation

Mean, Variance, Standard Deviation
	•	Mean((\mu)):
[
\mu = \frac{1}{N} \sum_{i=1}^N x_i.
]
	•	Variance((\sigma^2)):
[
\sigma^2 = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2.
]
	•	Std Dev((\sigma)):
[
\sigma = \sqrt{\sigma^2}.
]

Confidence Intervals
	•	Approximate range around a sample mean that’s likely to contain the true population mean.
	•	Example: 95% confidence interval in a normal distribution scenario.

Hypothesis Testing
	•	Null Hypothesis ((H_0)): Typically states “no difference” or “no effect.”
	•	p-value: Probability of observing a result at least as extreme as the one observed, assuming (H_0) is true.

	Real-World Use Case:
		•	A/B Testing in web analytics. You might test whether a new feature (Version B) leads to more user engagement than the old feature (Version A).

Calculus and Optimization

Derivatives & Gradients
	•	The derivative of a function (f(x)) measures how (f) changes as (x) changes.
	•	In multiple dimensions, the gradient (\nabla f(\mathbf{x})) is a vector of partial derivatives.

Chain Rule
	•	If (y = f(u)) and (u = g(x)), then
[
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}.
]
	•	In deep learning, we apply the chain rule across many nested layers.

Gradient Descent
	1.	Initialize parameters (\mathbf{w}).
	2.	Forward Pass: Compute predictions and loss.
	3.	Backward Pass: Compute gradients (\nabla L(\mathbf{w})).
	4.	Update:
[
\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla L(\mathbf{w}),
]
where (\alpha) is the learning rate.

	Graphs / Visualization
A parabolic loss function (L(w) = (w-5)^2) looks like a bowl. Gradient descent iteratively moves (w) closer to 5.

Practice Exercises (Prob/Stats/Calc)
	1.	Distribution Fit
	•	Generate random data from a normal distribution ((\mu=0), (\sigma=2)). Estimate mean and variance, compare to the ground truth.
	2.	Statistical Testing
	•	You have two sets of accuracy scores from different models. Conduct a t-test to see if one model is significantly better.
	3.	Partial Derivatives
	•	Let (f(x,y) = x^2 + 2xy + y^2). Compute (\frac{\partial f}{\partial x}) and (\frac{\partial f}{\partial y}).
	4.	Gradient Descent Example
	•	Manually implement gradient descent on a 1D function, like (L(w) = (w-3)^2 + 10). Track how (w) converges to 3.

Basic ML Models: Regression and Classification

Linear Regression

Given a dataset ({(\mathbf{x}^{(i)}, y^{(i)})}_{i=1}^N), where (\mathbf{x}^{(i)} \in \mathbb{R}^d) and (y^{(i)} \in \mathbb{R}), Linear Regression aims to learn:

[
\hat{y}^{(i)} = \mathbf{w}^T \mathbf{x}^{(i)} + b,
]
where (\mathbf{w}) is a weight vector, and (b) is a bias term.

Loss Function (MSE)
[
L(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^N \bigl(y^{(i)} - \hat{y}^{(i)}\bigr)^2.
]

Analytical Solution (Closed-Form)

[
\mathbf{w}^* = (X^T X)^{-1} X^T \mathbf{y},
]
where (X) is ((N \times d)), and (\mathbf{y}) is ((N \times 1)). Useful when (d) isn’t too large.

Gradient Descent Approach
	•	More feasible for large (d).
	•	Iteratively updates (\mathbf{w}) based on the gradient of MSE.

	Real-World Use Case
		•	Predicting House Prices: Input features: square footage, location, age of the house. Output: predicted price.

Logistic Regression

A binary classification model that predicts the probability of class “1” given (\mathbf{x}):

[
\hat{p} = \sigma(\mathbf{w}^T \mathbf{x} + b), \quad \text{where } \sigma(z) = \frac{1}{1 + e^{-z}}.
]

Loss Function (Cross-Entropy)

[
L(\mathbf{w}, b) = -\frac{1}{N}\sum_{i=1}^N \Bigl[y^{(i)} \log(\hat{p}^{(i)}) + (1-y^{(i)}) \log\bigl(1-\hat{p}^{(i)}\bigr)\Bigr].
]
	•	(\hat{p}^{(i)} \in [0,1]) is the probability of the positive class.
	•	Decision boundary: If (\hat{p}^{(i)} \geq 0.5), predict class 1; else class 0.

	Real-World Use Case
		•	Spam Detection: (\hat{p}) is the probability an email is spam, learned from labeled examples.

Overfitting and Regularization

Overfitting
	•	A model memorizes training data noise instead of generalizing.
	•	Symptom: Low training error but high validation/test error.

Regularization Methods
	1.	L2 (Ridge): Adds (\lambda |\mathbf{w}|^2) to the loss.
	2.	L1 (Lasso): Adds (\lambda |\mathbf{w}|_1) to the loss. Encourages sparse solutions.
	3.	Early Stopping: Halt training when validation performance stops improving.

	Visual Explanation
		•	L2 shrinks all weights but never exactly zeroes them. L1 can zero out some weights, effectively feature selection.

Practice Exercises (Basic ML)
	1.	Linear Regression
	•	Implement gradient descent for a 2D synthetic dataset. Plot the learned line. Compare with the closed-form solution.
	2.	Logistic Regression
	•	Use a small binary classification dataset (e.g., “above-average vs. below-average” house prices). Visualize the decision boundary.
	3.	Regularization Experiment
	•	Test L1 vs. L2 on a dataset with many features. Compare the number of zero-valued weights. Evaluate performance differences on a test set.

Summary and Further Reading

Key Takeaways
	1.	Linear Algebra: Underpins neural operations (vectors, matrices, SVD).
	2.	Probability & Statistics: Interpret model outputs, run significance tests, handle distributions.
	3.	Calculus: Core for gradient-based optimization (the backbone of neural network training).
	4.	Basic ML Techniques: Linear and logistic regression illustrate parameter fitting and highlight overfitting/regularization.

	Next Steps
In the next chapter, we’ll build upon these foundations to discuss neural networks, NLP preprocessing, and how data flows through early NLP pipelines.

Recommended Books/Resources
	•	Mathematics for Machine Learning by Deisenroth, Faisal, and Ong.
	•	Deep Learning by Goodfellow, Bengio, and Courville (see early math chapters).
	•	Andrew Ng’s Machine Learning Course on Coursera for further linear/logistic regression deep dives.

Chapter 1 Dictionary
	1.	Vector
	•	A 1D array of numbers. Used to represent data samples, model weights, or directions in space.
	2.	Matrix
	•	A 2D array of numbers (rows × columns). Fundamental for neural network operations.
	3.	Eigenvalue / Eigenvector
	•	(\mathbf{v}) is an eigenvector of (A) if (A\mathbf{v} = \lambda \mathbf{v}). (\lambda) is the eigenvalue.
	•	Key in PCA, dimensionality reduction, and analyzing transformations.
	4.	Singular Value Decomposition (SVD)
	•	A factorization (M = U\Sigma V^T). Used in data compression, noise reduction, matrix factorization.
	5.	Probability Distribution
	•	Describes how likely each outcome is for a random variable (e.g., Gaussian, Bernoulli).
	6.	Gradient Descent
	•	An iterative method to find (local) minima of a function by updating parameters in the opposite direction of the gradient.
	7.	Linear Regression
	•	Predicts a continuous output (y) from input (\mathbf{x}) via a linear model, minimizing MSE.
	8.	Logistic Regression
	•	A classification model outputting probabilities via the sigmoid function, minimizing cross-entropy.
	9.	Cross-Entropy Loss
	•	A measure of the distance between two probability distributions. In classification, used to compare predicted vs. true label distributions.
	10.	Regularization
	•	Adding penalties (L1/L2) or constraints to prevent overfitting and improve generalization.
	11.	Overfitting
	•	When a model memorizes training data noise rather than learning underlying patterns, leading to poor performance on unseen data.

	End of Chapter 1
You have laid the mathematical foundation for understanding deeper content in subsequent chapters, where we will explore neural architectures, Transformer mechanics, and ultimately Large Language Models. Be sure to tackle the practice exercises, as they will reinforce your newly gained mathematical insights in a practical manner.

