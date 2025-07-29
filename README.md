# C++ Nueral Network
A simple neural network implemented from scratch in C++14 with no external libraries or frameworks. This example demonstrates basic feedforward and backpropagation processes to output numerical predictions.

## Overview
This neural network performs two key operations:

* Feedforward: Passes input data through each layer to generate an output (prediction).

* Backpropagation: Adjusts weights based on prediction errors to improve future accuracy using gradient descent.

### Feedforward
Feedforward is the prediction phase where information moves in one direction through the network.

* Unidirectional Flow: Data moves from the input layer, through hidden layers, to the output layer.

* Layer Computation: Each layer multiplies inputs by weights, adds biases, and applies an activation function.

* Output: The final result is the network's prediction for the input data.

### Backpropagation
Backpropagation is the learning phase where the network updates its weights to minimize prediction errors.

* Error Calculation: The difference between the predicted and actual output is computed using a loss function.

* Gradient Descent: Calculates the gradient of the error with respect to each weight.

* Weight Updates: Weights are adjusted in the direction that reduces the error.

* Reverse Flow: The error is propagated backward from the output to the input layer, updating weights along the way.

## How to run:
```bash
  g++ -std=c++14 main.cpp
```

## Demo outout:
<img width="1195" height="891" alt="Screenshot 2025-07-28 at 2 43 08 PM" src="https://github.com/user-attachments/assets/56fd80ce-9f9e-41e9-98ef-c084e1e7708f" />
