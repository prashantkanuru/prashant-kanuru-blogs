# Earlier Note:

**I am going to try and cover how I vibe coded Automatic Differentiation and improved it iteratively, post that try and create an LLM-OS based on
my understanding of it and extend it to Agent Computer Interface, again through vibe coding. Extend this using Agentic Interface to extend it to some major AI libraries in python so one can extend them effortlessly**


## Vibe Coding Automatic Differentiation
This blog is an effort to understand Autograd aka computational implementation of back-propagation to implement gradient descent on parameters, the learning algorithm - gradient descent, the descent is implemented using back-propagation and computationally in frameworks like PyTorch and JAX it is implemented using Automatic Differentiation. Automatic Differentiation in turn depends upon usage of Jacobian Matrix and Computational graph.

## Jacobian:
___

The Jacobian matrix, $J_f$, is essentially the generalization of the derivative for a function that takes multiple inputs and produces multiple outputs 

**Definition**
For a vector valued function $f:\mathbb{R}^n \to \mathbb{R}^m$, which maps $n$ input variables $\mathbf{x} = [x_1, \dots, x_n]^T$ to $m$ output variables $\mathbf{f} = [f_1, \dots, f_m]^T$, the Jacobian matrix $J_f$ is an $m \times n$ matrix where the element in the $i$-th row and $j$-th column is the partial derivative of the $i$-th output with respect to the $j$-th input:
$$J_f = \frac{\partial\mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix}\frac{\partialf_1}{\partialx_1} & \frac{\partial f_1}{\partial f_1}{\partialx_2} & \cdots & \frac{\partial f_1}{\partial x_n}\\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n }\\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial f_m}{\partial x_2} & \cdots &\frac{\partial f_m}{\partial x_n}
\end{bmatrix}$$

### Vector Jacobian Product (VJP) - Reverse Mode (BackPropagation)
THe VJP Computes the product $\mathbf{x}^T = \mathbf{w}^T \cdot J_f$ 

**Interpretation:** The Jacobian describes the local linear approximation of the function. it tells you how much each output component changes with respect to a small change in each input component.

____

### Jacobian and Automatic Differentiation

|**Mode |**Operation** |**Product Computed** |**Cost** |**When it is efficient** |
|------ |-------------- |-------------------  |------- |------------------------ |
|**Forward Mode (JVP)** |Jacobian-vector Product |$J_f\cdot \mathbf{v}$ |$O(n \cdot C)$ |When $n \ll m$ (**Many outputs**, few inputs). |
|**Reverse Mode** |Vector-Jacobian Product |$\mathbf{v}^T \cdot J_f$ |$O(m \cdot C)$|When $m \ll n$ (**Few outputs, many inputs). | 


    Chat Question: What is Automatic Differentiation?
    Answer: Automatic Differentiation (AD) is a computational technique used to evaluate derivatives of functions efficiently and accurately.
    It is widely used in machine learning, optimization, and scientific computing. Unlike numerical differentiation (which approximates derivatives using
    finite differences) or symbolic differentiation (which manipulates algebraic expressions), AD computes derivatives by systematically applying the chain rule
    at a computational level.

    **Types of Automatic Differentiation:**
    1. **Forward Mode AD:**
      - Computes derivatives alongside function evaluation
      - Efficient when the function has fewer inputs and more outputs.
      - Works well for functions with scalar inputs and vector outputs
    2. **Reverse Model AD:**
      - Computes derivatives in a backward pass (used in backpropagation in deep learning)
      - Efficient when the function has many inputs and a few outputs (e.g. neural network training)
      - Commonly implemented in libraries like PyTorch, TensorFlow, and JAX.
    **Key Advantages:**
      - **Exact Derivatives:** Unlike finite differences, AD does not suffer from numerical instability
      - **Efficient Computation:** Reduces redundant computations and scales well.
      - **Automated Process:** No need to manually derive gradients.
    Would you like an example in Python using a library like PyTorch or JAX?


    Chat Question: What do you mean when you say: Automatic Differentiation applies chain rule at a computational level
    Answer: When I say **"Automatic Differentiation (AD) applies the chain rule at a computational level,"** I mean that AD systematically breaks down the
    computation of a function into elementary operations and then applies the **chain-rule** step-by-step during execution. This process ensures efficient
    and accurate derivative calculation.
    $y = f(x) = \sin\left(x^2\right)$

    **How AD Works Internally:**
    AD does not symbolically manipulate the function to find an algebraic derivative. Instead, it:
      1. Breaks down the function
    
    **Computational Example in Python (Using PyTorch):**
    ```
    import torch

    x=torch.tensor(2.0,requires_grad=True) # Track gradient
    y = torch.sin(x**2) #Function: y= sin(x**2)

    y.backward()
    print(x.grad)
    ```
  ### Observations on Vibe Coding:
  - One should add in different types of questions that they would want like asking it to explain why finite differences suffer from numerical instability.
  - The aim should be understand the control theory of the LLMs and possibly extend it to Gradient descent of prompts.
