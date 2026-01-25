---
layout: default
title: "Jacobian, VJP Tape and Computational Graph aka AutoGrad"
permalink: /automatic-differentiation-jacobian/
category: fundamentals
---
# Demystifying Automatic Differentiation through the lens of Jacobain: VJP Tape and Computational Graph

___

## 1. Introduction: The Language of Sensitivity

___

### What is Automatic Differentiation?
**Automatic Differentiation (AD)** is a set of techniques to evaluate the derivative of a function defined by a computer program. AD exploits the fact that every computer program defining a function (a neural network - compositional function, a physics simulation, or a simple loop), is executed as a sequence of **primitive operations** (addition, multiplication, exp, sin, etc.)

**AD** adds in the aspect of simultaneous application of chain-rule to these primitive operations overcoming the limitations of symbolic math and numerical approximations, making it the de-facto choice for gradient computation in deep-learning frameworks like PyTorch and JAX.

### Bridging the "Chain Rule" and the "Jacobian"

The fundamental objective of **AD** is to compute the sensitivity of a program's output to its input. Mathematically, this sensitivity is exactly what the **Jacobian** represents. 

### Viewing AD in terms of the Jacobian

Calculus in high-school or in an introductory 101 level is about a single variable but in case of Machine learning, it moves to "vector-to-vector" mappings. Combining the key ideas discussed till now, **AD can be viewed as a systematic way to compute the Jacobian by breaking down a complex program into a sequence of elementary operations**, thus every line of code in a PyTorch or JAX function represents a transformation (that is matrix operations, in a linear algebra sense, multiplying with a matrix to carry out a linear-transformation) and **AD** looks at these transformations as: "Each step having its own local jacobian (the sensitivity trapper aka multi-variable calculus, also called vector calculs) and if all these Jacobians are multiplied together, it leads to the creation of the global jacobian of the entire **program**".
In the next sub-section - I will be providing a small 101 on scalar vs vector valued functions.
___

#### Scalar vs. Vector Valued Functions
___

To understand the Jacobian, it is very helpful to understand the basic difference between scalar and vector valued functions. 

**Scalar Valued Functions: ($f: \mathbb{R}^n \to \mathbb{R}$)**
- Definition: These functions take a vector of inputs but return a **single number aka scalar output**.
- Example: A **Loss Function** in Deep Learning. It takes millions of weights (a vector) and returns a single "error" value (aka scalar output).
- Derivative: The derivative of a scalar function is called the **Gradient** ($\nabla f$). It is a vector that points in the direction of the steepest increase. 

**A Small Note**: Though the output of a scalar valued function is a scalar as in $\mathbb{R}$ but the input domain is still $\mathbb{R}^n$ and hence the gradient of the function will be a $\nabla f$ i.e. for a scalar function $f(x_1,x_2,\dots,x_n)$ is:
$$\nabla f = \left[\frac{\partial f}{\partial x_1},\frac{\partial f}{\partial x_2},\dots,\frac{\partial f}{\partial x_n} \right]$$      
