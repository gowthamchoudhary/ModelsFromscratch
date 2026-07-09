# Neural Network From Scratch

This project is a from-scratch implementation of a simple feedforward neural network for handwritten digit classification. The notebook builds the model using NumPy instead of deep learning frameworks, so the main goal is to understand the theory, maths, and training process behind neural networks.

The implementation is in [NeuralNetworkFromScracth.ipynb](NeuralNetworkFromScracth.ipynb).

## Project Overview

The model is trained on MNIST-style digit data where each image is a `28 x 28` grayscale image flattened into `784` input features. Each image belongs to one of `10` classes, representing digits from `0` to `9`.

The notebook covers:

- Loading the dataset from Google Drive in Google Colab
- Splitting the data into training and validation sets
- Normalizing pixel values from `0-255` to `0-1`
- Initializing weights and biases manually
- Implementing forward propagation
- Applying ReLU and softmax activation functions
- Converting labels into one-hot encoded vectors
- Implementing backpropagation
- Updating parameters using gradient descent
- Checking training and validation accuracy
- Visualizing a prediction on a validation image

## Model Architecture

The neural network has three main layers:

```text
Input layer:   784 neurons
Hidden layer:   10 neurons, ReLU activation
Output layer:   10 neurons, softmax activation
```

Each input image is represented as a column vector:

```text
X shape = (784, number_of_examples)
```

The parameters used by the model are:

```text
W1 shape = (10, 784)
B1 shape = (10, 1)
W2 shape = (10, 10)
B2 shape = (10, 1)
```

## Maths Behind the Network

### Forward Propagation

The first layer computes a linear transformation:

```text
Z1 = W1 . X + B1
```

Then ReLU is applied:

```text
A1 = ReLU(Z1)
ReLU(x) = max(0, x)
```

The second layer computes:

```text
Z2 = W2 . A1 + B2
```

The output probabilities are calculated using softmax:

```text
A2 = softmax(Z2)
softmax(z_i) = exp(z_i) / sum(exp(z_j))
```

The predicted digit is the class with the highest probability:

```text
prediction = argmax(A2)
```

### One-Hot Encoding

The actual labels are converted into one-hot encoded vectors. For example, if the correct digit is `3`, the label becomes:

```text
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

This makes it easier to compare the model's output probabilities with the correct class.

### Backpropagation

The output layer error is:

```text
dZ2 = A2 - one_hot_Y
```

Gradients for the second layer:

```text
dW2 = (1 / m) . dZ2 . A1.T
dB2 = (1 / m) . sum(dZ2)
```

The hidden layer error is passed backward through `W2` and multiplied by the derivative of ReLU:

```text
dZ1 = W2.T . dZ2 * ReLU'(Z1)
```

Where:

```text
ReLU'(z) = 1 if z > 0
ReLU'(z) = 0 if z <= 0
```

Gradients for the first layer:

```text
dW1 = (1 / m) . dZ1 . X.T
dB1 = (1 / m) . sum(dZ1)
```

### Gradient Descent

The parameters are updated using gradient descent:

```text
W1 = W1 - learning_rate * dW1
B1 = B1 - learning_rate * dB1
W2 = W2 - learning_rate * dW2
B2 = B2 - learning_rate * dB2
```

In the notebook, the model is trained with:

```text
learning_rate = 0.1
iterations = 1000
```

## How to Run

1. Open the notebook in Google Colab.
2. Mount Google Drive.
3. Place the training CSV file at:

```text
/content/drive/MyDrive/train.csv
```

4. Run the notebook cells from top to bottom.
5. Watch the training accuracy printed every 20 iterations.
6. Check the final validation accuracy.
7. View a sample validation image with its predicted and actual label.

## Requirements

The notebook uses:

- Python
- NumPy
- Pandas
- Matplotlib
- Google Colab

Install the Python libraries locally with:

```bash
pip install numpy pandas matplotlib
```

## What I Learned

This project helps explain how a neural network works internally:

- How image data is represented numerically
- How weights and biases transform input data
- Why activation functions are needed
- How softmax turns raw scores into probabilities
- How loss gradients flow backward through the network
- How gradient descent improves predictions over time
- How model accuracy is measured on unseen validation data

Instead of using libraries like TensorFlow or PyTorch, this project focuses on building the main neural network steps manually to understand the maths behind deep learning.

## Notes

The notebook file name currently uses `Scracth` instead of `Scratch`. The project still runs normally, but the spelling can be renamed later if needed.
