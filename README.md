# Neural Network from Scratch – MLP on MNIST, CIFAR‑10 & Diabetes

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)](https://www.python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)](https://jupyter.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?style=flat&logo=numpy)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5%2B-11557C?style=flat)](https://matplotlib.org)
[![Scikit‑learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-F7931E?style=flat&logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Overview

This repository contains **three from‑scratch implementations of a Multilayer Perceptron (MLP)** – a classic feed‑forward neural network – applied to three very different datasets:

1. **MNIST** – handwritten digit recognition (28×28 grayscale images)  
2. **CIFAR‑10** – object recognition in small colour photographs (32×32 RGB)  
3. **Diabetes (Pima Indians)** – tabular medical data for binary classification

Each notebook builds the entire neural network **without high‑level deep‑learning frameworks** (no TensorFlow, PyTorch, Keras). Only `numpy` is used for matrix operations, and `matplotlib` for visualisations. The goal is to understand what happens *inside* a neural network – from weight initialisation to back‑propagation – and to observe how the same core model behaves on data of different modalities.

---

## 🧠 What is an MLP? (Multilayer Perceptron)

A **Multilayer Perceptron (MLP)** is the simplest form of an artificial neural network. It consists of **an input layer, one or more hidden layers, and an output layer**, with each layer fully connected to the next[reference:0].

### Key Components
<img width="441" height="239" alt="nodeNeural" src="https://github.com/user-attachments/assets/76224e6b-c3be-475b-9cf6-598216fd71e0" />


#### 1. Input Layer
The first layer receives the raw data. Each neuron in this layer corresponds to one feature:
- For **images** (MNIST, CIFAR‑10), each pixel is a feature. A 28×28 grayscale image gives 784 input neurons; a 32×32 RGB image gives 3 072 (32×32×3).
- For **tabular data** (Diabetes), each column is a feature (e.g., glucose level, BMI, age).

#### 2. Hidden Layers
These are the computational core of the network. Each hidden layer performs two operations:

1. **Linear transformation**:  
   `Z = X · W + b`  
   where `X` is the input from the previous layer, `W` is the weight matrix, and `b` is the bias vector[reference:1].

2. **Non‑linear activation**:  
   The output `Z` is passed through an activation function (e.g., **ReLU**, **Sigmoid**, **Tanh**). Without non‑linearity, the entire network would collapse into a single linear transformation – no matter how many layers are stacked[reference:2].

> 💡 **Why layers?**  
> Each successive layer learns increasingly abstract representations. In image tasks, early layers detect edges, middle layers detect shapes, and deeper layers detect whole objects.

#### 3. Output Layer
The final layer produces the prediction:
- **Multi‑class classification** (MNIST, CIFAR‑10): uses **Softmax** activation, which converts raw scores into a probability distribution over all classes. The number of neurons equals the number of classes (10).
- **Binary classification** (Diabetes): uses **Sigmoid** activation, which squashes the output to [0, 1] – the probability of the positive class.

#### 4. Loss Function
The network’s performance is measured by a loss function:
- **Categorical Cross‑Entropy** for MNIST and CIFAR‑10.
- **Binary Cross‑Entropy** for Diabetes.

The loss tells us how far the predictions are from the true labels, and the entire training process aims to minimise it.

#### 5. Back‑propagation & Gradient Descent
This is how the network *learns*:

1. **Forward pass**: input flows through the network to produce a prediction.
2. **Loss calculation**: the prediction is compared with the true label.
3. **Backward pass**: the gradient of the loss with respect to every weight and bias is computed using the chain rule.
4. **Update**: weights and biases are nudged in the opposite direction of the gradient, scaled by a **learning rate**.

Mathematically, for a weight w is `w = w-η.(∂L / ∂w)`
where `η` is the learning rate.

This cycle repeats for many **epochs** until the loss stabilises[reference:3].

---

## 📦 Datasets at a Glance

### ✍️ MNIST – Handwritten Digits

| Property | Value |
|---|---|
| Total images | 70 000 |
| Training set | 60 000 |
| Test set | 10 000 |
| Image size | 28 × 28 pixels, grayscale |
| Number of classes | 10 (digits 0–9) |
| Input dimension | 784 (28 × 28) |

The MNIST database is often called the *“Hello, World!”* of machine learning[reference:4]. It was created by Yann LeCun and colleagues from NIST’s original datasets. All digits are size‑normalised and centred in a fixed‑size frame. Despite its simplicity, MNIST remains an excellent benchmark for testing classification algorithms and is large enough to give statistically meaningful results without requiring heavy hardware.

**Why MNIST?**  
Because it is a clean, well‑understood dataset, it lets us verify that our from‑scratch MLP is correctly implemented before moving to more challenging data.

---

### 🖼️ CIFAR‑10 – Tiny Colour Images

| Property | Value |
|---|---|
| Total images | 60 000 |
| Training set | 50 000 |
| Test set | 10 000 |
| Image size | 32 × 32 pixels, RGB colour |
| Number of classes | 10 |
| Input dimension | 3 072 (32 × 32 × 3) |

CIFAR‑10 was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton[reference:5]. The 10 classes are: **airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck** – with 6 000 images per class.

CIFAR‑10 is significantly harder than MNIST for an MLP because:
- Objects vary in pose, lighting, and background.
- Colour triples the number of input features (3 072 vs. 784).
- The small 32×32 resolution makes fine details challenging to discriminate.

> 🏆 **Baseline results** reported with a convolutional neural net are ~18 % test error without data augmentation and ~11 % with augmentation[reference:6]. An MLP is not expected to match these numbers, but seeing *how close* it can get is the instructive part.

---

### 🩺 Pima Indians Diabetes Dataset

| Property | Value |
|---|---|
| Total samples | 768 |
| Features | 8 |
| Target | 0 (non‑diabetic) / 1 (diabetic) |
| Input dimension | 8 |

This dataset originates from the National Institute of Diabetes and Digestive and Kidney Diseases. It contains diagnostic measurements for female patients of Pima Indian heritage (at least 21 years old) living near Phoenix, Arizona[reference:7].

| # | Feature | Description |
|---|---|---|
| 1 | Pregnancies | Number of times pregnant |
| 2 | Glucose | Plasma glucose concentration (2‑hour oral glucose tolerance test) |
| 3 | Blood Pressure | Diastolic blood pressure (mm Hg) |
| 4 | Skin Thickness | Triceps skinfold thickness (mm) |
| 5 | Insulin | 2‑hour serum insulin (μU / ml) |
| 6 | BMI | Body mass index (weight / height²) |
| 7 | Diabetes Pedigree | Diabetes pedigree function (a genetic risk score) |
| 8 | Age | Age in years |

Unlike the image datasets, the Diabetes data is **tabular** and **very small** (only 768 samples). This brings different challenges: overfitting, missing values (several features contain zeros that are physiologically impossible), and the need for careful normalisation.

---
# Neural Network from Scratch

A collection of Jupyter notebooks that implement **Multi-Layer Perceptrons (MLPs) from scratch using NumPy** and apply them to three different datasets:

- Handwritten digit classification using the MNIST dataset
- Image classification using the CIFAR-10 dataset
- Binary classification using the Pima Indians Diabetes dataset

This repository is designed to provide a deep understanding of how neural networks work internally by implementing:

- Forward propagation
- Backpropagation
- Gradient descent
- Loss computation
- Weight updates

Rather than relying on high-level frameworks, these notebooks build every component manually to demonstrate the underlying mechanics of neural networks.

---

## 📓 Notebook 1: MLP on MNIST

**File:** `MLP_using_MNIST.ipynb`

### Dataset Overview

The MNIST dataset contains:

- 70,000 grayscale images of handwritten digits
- Image size: 28 × 28 pixels
- 10 output classes (digits 0–9)

### Preprocessing Steps

1. Load the dataset using `tensorflow.keras.datasets` or `sklearn`.
2. Flatten each image from `28 × 28` to a `784-dimensional` vector.
3. Normalize pixel values from `[0, 255]` to `[0, 1]`.
4. One-hot encode labels.

Example:

```text
Digit 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

### Network Architecture

```text
Input Layer:    784 neurons
Hidden Layer 1: 128 neurons (ReLU)
Hidden Layer 2:  64 neurons (ReLU)
Output Layer:    10 neurons (Softmax)
```
<img width="876" height="541" alt="1MmrWSRkddKWmY7uAnp6DgQ" src="https://github.com/user-attachments/assets/da8a4056-df55-4e12-a588-a52bd1cbb553" /><img width="1182" height="537" alt="21565642-fbe3-4762-889f-de57d52ef8e8" src="https://github.com/user-attachments/assets/105a9b11-f46d-40dd-b558-6f824cbbd6ef" />


### Training Configuration

| Hyperparameter | Typical Value |
| -------------- | ------------- |
| Learning Rate  | 0.01 – 0.1    |
| Epochs         | 50 – 200      |
| Batch Size     | 32 – 128      |
| Loss Function  | Categorical Cross-Entropy |

### Results and Observations

- Achieves approx ~ 98% Accuracy
- Minimal overfitting due to large dataset size

  <img width="576" height="455" alt="download" src="https://github.com/user-attachments/assets/0bc5a68f-0e8d-4cb7-8093-a27cd2064ddf" />


### Key Learnings

- Basic MLPs perform extremely well on clean datasets
- Data normalization is critical
- Learning rate significantly affects convergence


---

## 📓 Notebook 2: MLP on CIFAR-10

**File:** `MLP_using_CIFAR10.ipynb`

### Dataset Overview

The CIFAR-10 dataset contains:

- 60,000 color images
- Image size: 32 × 32 × 3
- 10 object classes

### Preprocessing Steps

1. Load the CIFAR-10 dataset.
2. Flatten each image into a `3072-dimensional` vector.
3. Normalize pixel values to `[0, 1]`.
4. One-hot encode labels.

### Network Architecture

```text
Input Layer:    3072 neurons
Hidden Layer 1: 1024 neurons (ReLU)
Hidden Layer 2:  512 neurons (ReLU)
Hidden Layer 3:  256 neurons (ReLU)
Output Layer:     10 neurons (Softmax)
```

### Training Configuration

| Hyperparameter | Typical Value |
| -------------- | ------------- |
| Learning Rate  | 0.001 – 0.01  |
| Epochs         | 100 – 300     |
| Batch Size     | 64 – 256      |
| Regularization | Dropout or L2 (optional) |

<img width="556" height="455" alt="download" src="https://github.com/user-attachments/assets/3d370707-2830-40c2-a8f9-bce0279678b9" />
<img width="485" height="451" alt="b3548d95-0446-4300-8f9a-ebac9b688723" src="https://github.com/user-attachments/assets/6aff3fb7-a161-423b-8c1d-81873e777d5c" />


### Results and Observations

- Typical test accuracy: **45–55%**
- Significant overfitting can occur
- Training is much slower than MNIST
- Performance is far below CNNs, which can exceed 90%

  <img width="1182" height="537" alt="21565642-fbe3-4762-889f-de57d52ef8e8" src="https://github.com/user-attachments/assets/5eb7b67a-ecdb-467d-952f-914150cebdd5" />


### Key Learnings

- MLPs ignore spatial structure
- CNNs are better suited for image tasks
- Regularization is essential
- Hyperparameter tuning has major impact

<img width="700" height="547" alt="cfb0258c-339e-4f14-8644-0f86b089d074" src="https://github.com/user-attachments/assets/094e7ed4-6997-4aa9-8056-9c7cfa6a95fc" />



---

## 📓 Notebook 3: MLP on Diabetes Dataset

**File:** `MLP_using_Diabetes.ipynb`

### Dataset Overview

The Pima Indians Diabetes dataset contains:

- 768 patient records
- 8 medical features
- Binary output:
  - `0` = Non-diabetic
  - `1` = Diabetic

### Preprocessing Steps

1. Load the dataset from CSV or `sklearn`.
2. Replace biologically impossible zeros with `NaN`.
3. Impute missing values using mean or median.
4. Standardize features.
5. Split into training and testing sets.

### Network Architecture

```text
Input Layer:      8 neurons
Hidden Layer 1:  16 neurons (ReLU)
Hidden Layer 2:   8 neurons (ReLU)
Output Layer:     1 neuron (Sigmoid)
```

### Training Configuration

| Hyperparameter | Typical Value |
| -------------- | ------------- |
| Learning Rate  | 0.001 – 0.01  |
| Epochs         | 100 – 500     |
| Batch Size     | 16 – 64       |
| Loss Function  | Binary Cross-Entropy |
| Metrics        | Accuracy, Precision, Recall |

### Results and Observations

- Typical test accuracy: **70–80%**
- Overfitting is common
- Standardization is essential
- Recall and precision are more meaningful than accuracy alone

<img width="556" height="455" alt="download" src="https://github.com/user-attachments/assets/43d88b17-1709-4c4d-8c44-9388ed276bf7" />


### Key Learnings

- Tabular data presents different challenges than image data
- Feature scaling is mandatory
- Early stopping helps prevent overfitting
- Medical applications require careful metric selection

---

## 🧪 Lessons Learned Across All Projects

### 1. Data Preprocessing Is Half the Battle

Proper normalization, standardization, and missing-value handling are critical to successful training.

### 2. Architecture Must Match the Problem

- **MNIST:** Simple MLP performs very well
- **CIFAR-10:** MLP struggles due to lack of spatial awareness
- **Diabetes:** Small network avoids overfitting

### 3. Hyperparameters Matter

Learning rate, batch size, and architecture depth can dramatically change performance.

### 4. Building from Scratch Deepens Understanding

Implementing all calculations manually makes the training process much clearer.

### 5. One Model Does Not Fit All

Different datasets require different preprocessing strategies, architectures, and evaluation metrics.

---

## 🚀 How to Run the Notebooks

### 1. Clone the Repository

```bash
git clone https://github.com/shouryashah05/NeuralNetwork-from-scratch.git
cd NeuralNetwork-from-scratch
```

### 2. Install Dependencies

```bash
pip install numpy matplotlib scikit-learn jupyter
```

> Optional: Install `tensorflow` or `keras` if needed to download MNIST and CIFAR-10 datasets.

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open and Run Any Notebook

- `MLP_using_MNIST.ipynb`
- `MLP_using_CIFAR10.ipynb`
- `MLP_using_Diabetes.ipynb`

---

## 📜 License

[MIT License](https://opensource.org/license/MIT)

Copyright (c) 2026 Shourya Shah 


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 🙏 Acknowledgements

- Yann LeCun and collaborators for the MNIST dataset
- Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton for CIFAR-10
- National Institute of Diabetes and Digestive and Kidney Diseases for the Pima Indians Diabetes dataset
- The open-source community for countless educational resources

---

## ⭐ Support This Project

If you found this repository helpful or learned something new, please consider giving it a **star ⭐** — it helps others discover it and motivates me to keep building and sharing!

> *“A star a day keeps the bugs away!”* 🐛✨

---

Built with curiosity, NumPy, and a lot of print statements.

**Happy Learning!**
