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

### Defining the Jacobian Matrix: The "Master Map"
The Jacobian ($J$) is a matrix that collects every possible first-order partial derivative of a vector-valued function. If the function $f$ maps $n$ inputs to $m$ outputs, the Jacobian is an $m \times n$ matrix:

$$J = \begin{bmatrix}
\frac{\partial y_1}{\partialx_1} & \cdots & \frac{\partial y_1}{\partial x_n}\\
\vdots & \ddots & \vdots \\
\frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n} 
\end{bmatrix}$$
Each Row ($i$): Represents how the $i$-th output changes with respect to all inputs (this is the gradient of $y_i$).
Each Column ($j$): Represents how all outputs change with respect to a "nudge" in the $j$-th input.

### Double Click on AD_Jacobian: Bridging the "Chain Rule" and the "Jacobian"

As stated above the fundamental objective of AD is to compute the sensitivity of a program's output to its input and mathematically **Jacobian** captures this sensitivity.

**1. Decomposing the Program into "Local" Jacobians**
Assuming a function $y = f(x)$ has multiple intermediate steps or function as in a composition as in the case of a neural network. So AD sees it as a sequence of intermediate steps.
$$x \to z_1 \to z_2 \to \dots \to y$$
Each step $z_{i} \to z_{i+1}$ is a vector-valued function. Therefore, each step has its own Local Jacobian ($J_i$), so if $z_1$ is a layer in a network, $J_1$ represents how that specific layer's outputs change with respect to its inputs.

**2. The Chain Rule as a compositional operation**:
The "Grand Objective" is to find the Global Jacobian ($J_{total}$), which picks up the final output $y$ changes with respect to the composed inputs $x$. Applying the Chain Rule, the Global Jacobian is simply the product of all the local Jacobians:
$$J_{total} = J_k \cdot J_{k-1} \cdot \dots \cdot J_1 $$

**3. The Choice: How to Multiply?**
Choice of multiplication is where the mode of AD **Forward** or **Reverse** gets enacted. This enactment of mode of AD uses the **associative** property of matrix.

- **Forward Mode (JVP)**: Multiply the matrices from right to left (starting from the input).
$$(J_k\cdot(J_{k-1}\cdot(J_1\cdot\mathbf{v})))$$

- **Reverse Mode (VJP)**: Multiply the matrices from left to right (starting from the output).

$$((\mathbf{v}^T \cdot J_k)\cdot J_{k-1})\cdot J_1$$
This computes the Vector-Jacobian Product.

**Summary**: In AD, the **Jacobian is the goal** and the **Chain Rule is the path**. AD looks at the function as a "Graph of Jacobians" and multiplication of the local Jacobians together AD is able to map in the forward mode how a tiny change in input will lead to a change in output and in the reverse mode how a tiny change in output can be mapped to change in input (aka parameters).

___

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

    Each component quantifies the change in scalar output with respect to the change in it. Combining these components gives a vector and thus the direction to move to bring about maximum change in the scalar output.
        - **Connection to Jacobian**: The Jacobian of the scalar function by definition of Jacobian is a row vector ($1 \times n$).
            - Jacobian acts as a linear map in terms of a which captures the change in the scalar output for a small "nudge" in the input space $\mathbb{R}^n$.

            $$\text{Change in Output (Scalar)} = \text{Jacobian(Row Vector)} \times \text{Nudge(Column Vector)} $$
    As a conclusion, even though a scalar function returns a single number, by the virtue of the domain being $\mathbb{R}^n$ it depends on many inputs and the **Gradient** captures the sensitivity of the scalar w.r.t all the inputs and in the language of **Jacobians** this gradient is just a Jacobian with only one row.

**Vector Valued Functions**($f:\mathbb{R}^n \to f:\mathbb{R}^m$)
    - **Definition**: These functions take a vector and return a **vector**
    - **Example**: A **Hidden Layer** in a neural network. It takes an input vector (features) and outputs another vector (activations).
    - **Derivative**: The derivative of a vector-valued function is the **Jacobian Matrix**
___

## 2. The "Curse of Dimensionality" in AD - why we need JVP and VJP tapes:
___

In a theoretical sense, the Jacobian is the complete "Master Map" of a function's sensitivity. However, in deep learning, it becomes computationally unoptimal to untractable to compute the full matrix.

**Why not?**
Let us take a standard transformation layer:
Inputs ($n$): 1 million parameters
Outputs ($m$): 1 million activations
The Jacobian Size: $10^6 \times 10^6 = 1,000,000,000,000$ elements (1 Trillion!).
Computing and storing this would require terabytes of memory for just a single layer. This is the "Curse of Dimensionality" in AD: while the Jacobian is mathematically elegant, it is computationally catastrophic. 
This is why JVPs and VJPs exist. They allow us to compute the effect of the Jacobian (the product) without ever actually building the massive itself.