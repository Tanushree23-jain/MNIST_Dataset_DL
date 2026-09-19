# MNIST Handwritten Digit Classification

A simple feedforward neural network built with **TensorFlow/Keras** to classify handwritten digits (0–9) from the MNIST dataset. This project was built as a deep learning assignment covering the full pipeline: data exploration, model design and training, evaluation, visualization, and an experiment comparing model configurations.

## 📌 Overview

- **Dataset:** [MNIST](http://yann.lecun.com/exdb/mnist/) — 60,000 training images and 10,000 test images of handwritten digits (28×28 grayscale)
- **Framework:** TensorFlow / Keras
- **Task:** Multi-class classification (10 classes: digits 0–9)
- **Model type:** Fully-connected (dense) neural network

## 🧠 What This Notebook Does

1. **Load and explore the dataset** — loads MNIST, inspects the shapes of the training/test sets, and displays a sample image for each digit class (0–9).
2. **Preprocess the data** — normalizes pixel values to the 0–1 range, flattens each 28×28 image into a 784-length vector, and one-hot encodes the labels.
3. **Build, compile, and train a neural network** — a `Dense(128) → Dense(128) → Dropout(0.25) → Dense(10, softmax)` architecture, trained with the Adam optimizer for 10 epochs.
4. **Evaluate the model** — reports test loss and test accuracy on unseen data, and visualizes results with a confusion matrix.
5. **Visualize training behavior** — plots training vs. validation accuracy and loss across epochs to check for overfitting.
6. **Test on sample images** — displays 5 test images with their predicted vs. true labels, plus the model's most confident misclassifications.
7. **Run a controlled experiment** — trains a second model identical to the first except for one change (neurons per hidden layer: 128 → 256), then compares the two models' performance side by side.

## 📊 Results

| Model | Hidden Layer Size | Parameters | Test Accuracy | Test Loss |
|---|---|---|---|---|
| Baseline | 128 neurons/layer | 118,282 | 97.67% | 0.0727 |
| Experiment | 256 neurons/layer | 269,322 | 98.12% | 0.0654 |

Doubling the hidden layer width improved test accuracy by ~0.45 percentage points, at the cost of roughly 2.3× more parameters and slower training per epoch.

## 🛠️ Tech Stack

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn (for the confusion matrix)

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Install the dependencies:
   ```bash
   pip install tensorflow keras numpy matplotlib seaborn scikit-learn
   ```
3. Open the notebook and run all cells:
   ```bash
   jupyter notebook mnist_digit_classifier.ipynb
   ```
   (or open it directly in [Google Colab](https://colab.research.google.com/))

## 📁 Repository Structure

```
├── mnist_digit_classifier.ipynb   # Main notebook: data, model, training, evaluation, experiment
└── README.md                      # Project overview (this file)
```

## 📚 What I Learned

This project helped me understand:
- How to preprocess image data for a fully-connected neural network
- How dropout and validation splits help detect and reduce overfitting
- How to read training/validation curves to judge whether a model is learning well
- How isolating a single hyperparameter change (and keeping everything else fixed) lets you measure its true effect on model performance

## 🙋 Author

Built as part of a deep learning assignment on image classification with neural networks.

---
*Feel free to fork this repo, experiment with the architecture, or try adding a CNN for comparison!*
