---
layout: default
title: "Vibe Coding: Automatic Differentiation"
permalink: /vibe-coding-autodiff/
category: innovations
---

**I am going to try and cover how I vibe coded Automatic Differentiation and improved it iteratively, post that try and create an LLM-OS based on my understanding of it and extend it to Agent Computer Interface, again through vibe coding. Extend this using Agentic Interface to extend it to some major AI libraries in python so one can extend them effortlessly.**

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
    ```python
    import torch

    x=torch.tensor(2.0,requires_grad=True) # Track gradient
    y = torch.sin(x**2) #Function: y= sin(x**2)

    y.backward()
    print(x.grad)
    ```

### Observations on Vibe Coding

- One should add in different types of questions that they would want like asking it to explain why finite differences suffer from numerical instability.
- The aim should be understand the control theory of the LLMs and possibly extend it to Gradient descent of prompts.
