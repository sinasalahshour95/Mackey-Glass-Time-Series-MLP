# Mackey-Glass-Time-Series-MLP
This project implements several **custom neural network algorithms from scratch in MATLAB** to predict the next value of a nonlinear chaotic time series — the **Mackey–Glass system**

## Problem Definition

The goal of this project is to predict  x(t+1),  given three previous samples:  

[x(t), x(t-1), x(t-2)]


The **Mackey–Glass** time-delay differential equation is defined as:

dx(t)/dt = -0.1·x(t) + 0.2·x(t - τ) / (1 + (x(t - τ))¹⁰)

where:

- τ = 17
- x(18) = 1.2
- tₘₐₓ = 1100
- x(t) is defined for t < 18

## Implemented Neural Network Algorithms

All algorithms are implemented **manually in MATLAB**, without using built-in training functions such as `trainlm` or `train`.

| Algorithm | Description |
|------------|--------------|
| **Conventional MLP** | Standard one-hidden-layer network trained using basic backpropagation. |
| **Emotional Learning** | Incorporates previous error into the weight update using an emotional coefficient. |
| **Adaptive Learning Rate** | Dynamically adjusts the learning rate during backpropagation. |
| **Levenberg–Marquardt (LM)** | A second-order optimization method that combines Gauss–Newton and gradient descent techniques. |

## Project Structure

 - Mackey_Glass.m            # Generate the Mackey–Glass time series and initial weights
 - MLP.m                     # One-hidden-layer MLP (standard backprop)
 - EMLP.m                    # Neural network with emotional learning
 - AdaptiveMLP.m             # Adaptive learning rate implementation
 - Levenberg_Marquardt.m     # Levenberg–Marquardt implementation

