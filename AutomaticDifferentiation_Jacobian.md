---
layout: default
title: "Jacobian, VJP Tape and Computational Graph aka AutoGrad"
permalink: /automatic-differentiation-jacobian/
category: fundamentals
---
# Demystifying Automatic Differentiation: The Lens of Jacobian, VJP Tape and Computational Graph

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
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n}\\
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

<div id="ad-animation-container"></div>

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

$$ \text{Change in Output (Scalar)} = \text{Jacobian(Row Vector)} \times \text{Nudge(Column Vector)} $$
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
This is why JVPs and VJPs exist. They allow us to compute the effect of the Jacobian (the product) without ever actually building the massive matrix itself.

___

## 3. Why JVPs and VJPs are considered a Shortcut - The Math vs. Engineering View - Tape & Pullback Functions
___

This question an obvious one we need to consider the computational costs, Jacobian is a full map, but in most cases, all that is needed is the knowledge of how a specific "signal" propagates through the network.

___

**1. The Shortcut Logic: Products over Matrices**
In a standard computer program, you have a sequence of functions: $x \to f_1 \to f_2\to \dots \to f_k \to y$ .
By the Chain Rule, the full Jacobian is:
$$J = J_k \cdot J_{k-1} \dots J_1$$
The "Expensive" Way: Multiply all these matrices together to get $J$, then multiply by a vector $v$.
The "Shortcut" Way: Treat each local Jacobian as a linear operator. Instead of materializing the matrix $J_1$ on a vector.
Forward Mode AD: The JVP Shortcut
Forward mode compute $Jv$. It tracks the "tangent" (velocity) of each intermediate variable as the computation moves forward.

How it works: It starts with a seed vector $v$ (usually a one-hot vector representing the input that is the target of interest). At each step $i$, it computes:
$$v_{i} = J_i\cdot v_{i-1}$$
The Shortcut: $J_i$ is never stored instead what is computed is the result of the product $v_{i}$.
Complexity: The complexity (Time) is, if there are $N$ inputs, the forward pass must be run $N$ times to get the full Jacobian.

___

**2. Reverse Mode AD: The VJP Shortcut**
Reverse mode computes $v^T J$, the core or heart of the learning process in deep learning as this enables/carries out the gradient of a scalar (like loss) with respect to every single parameter in one single backward pass.
How it works: First a "Forward Pass" is carried out to compute and store the values of all intermediate variables. The learning journey or the backward (backward because it starts from the loss) starts from the output ($v = 1$ for a scalar loss):
$$g_{i-1} = v^T_{i}\cdot J_i $$
The Shortcut: Each step is a Vector-Jacobian Product. It is just a "Pulling back" of the gradient from the output towards the input.
Complexity: (Time), irrespective of the number of inputs $N$, if there is 1 scalar output (which is the case as it is "a" loss -  "a" scalar), only one backward pass is needed.

___

**3. Why these are called "Shortcuts"**
The Jacobian $J$ for a layer with 1,000 inputs and 1,000 outputs  (and this is a very simple one for that matter the inputs and output can get multiple "X's" of these) has 1,000,000 entries

- **Memory**: A JVP or VJP only require storing the vectors (size 1,000), this is done in PyTorch as ctx, not the matrix
- **Computation**: Modern hardware (GPUs/ASICs) are faster at performing a sequence of Vector-Matrix or Vector-Vector operations than they are at performing Matrix-Matrix multiplications of massive, often sparse, Jacobians.
___

**4. Now, how do the frameworks actual bring the shortcuts into usage**

The usage of the shortcuts is what helps answer possible doubts or questions like: **If we do not even create the Jacobian, how does the computer know how to multiply by it?**

Put in another way, if we are not building the Jacobian matrix, how does the framework "know" how to differentiate or else the derivative or gradient function?
This is handled in two distinct phases during the execution of the code.

**Phase 1: The Primary Computation (The "What")**
The framework first executes the primitive operation to get the output.
Example: $y = \sin(x)$
The computer calculates the sine of the input and stores the result $y$. This is the standard forward pass we are all familiar with.

**Phase 2: The Sensitivity Recording (The "How")**
This is where the implementation splits depending on the `Automatic Differentiation Mode` (Forward or Reverse). The framework `"Tapes"` a specialized function to the computational node.

For Forward AD (Storing the JVP):

If the framework is in "Forward Mode", it does not wait for a backward pass. It immediately computes the **Jacobian-Vector Product** alongside the primal value.

- **Input**: $x$ and a "tangent" vector $v$
- **Calculation**: $y = \sin(x)$ and $\dot{y} = \cos(x) \cdot v$.
- **Result**: The "tangent" $\dot{y}$ is moved to the next operation immediately.

For Reverse AD (Storing the VJP/Pullback):

In Reverse Mode (PyTorch or JAX's `vjp`), the framework cannot compute the derivative yet because it does not have the "incoming gradient" from the loss. Instead, it stores a **Pullback Function**.

- **Forward Pass**: $y = \sin(x)$
- **The "Tape" Record**: The framework saves the input $x$ (because it is needed for the derivative) and a pointer to a specialized function `lambda grad: grad * cos(x)`.
- **Waiting**: This function sits on the "Tape" until the backward pass begins.

**Key Distinction**: In Forward AD, we compute and store **values** (tangents). In Reverse AD, we compute and store **functions** (pullbacks).

___

**Finally: Why Flattening is an Illusion**
When we see the math $J^T\mathbf{v}$, we except to see a matrix, but the "Tape" just shows us that $J^T\mathbf{v}$ is just a recipe.
- The Math View: $J$ is a matrix of partial derivatives. To get $J^T\mathbf{v}$, the matrix needs to be arranged, transposed and multiplied.
- The Engineering View: $J^T\mathbf{v}$ is a specific **Pullback Function** for a specific operation.

For a matrix multiplication $Y = WX$, the VJP for $W$ is simply $\mathbf{v}X^T$. The framework does not "flatten" the weights into a Jacobian; it just executes that specific transposed multiplication.

___

## 4. How the pullback functions are executed using a data structure called `Toposort`

Once the `"Tape"` is full of these pullback functions, the framework needs to execute them. It cannot go in any random order; it must follow the **Reverse Topological Order**
- The **Loss Node**: Gradient of 1.0 is the starting point.
- **Topological Traversal**: The framework looks at the last operation on the tape, grabs its stored `pullback_fn`, passes in the current gradient, and gets a new gradient.
- **Accumulation**: If a variable was used in multiple places (like a weight used in two different layers), the gradients are **summed** at that node.

### A Double-Click on the process in PyTorch: `ctx` and the Dynamic Graph

<div id="pytorch-pointer-animation-container"></div>

In PyTorch, every operation that involves a tensor with `requires_grad = True` creates a **Node** in the computational graph. This node is an instance of `torch.autograd.Function`.

- **The `ctx` (The Context Object)**: The `ctx` is the local memory for a specific operation. It acts as the "bridge" between the forward pass and the backward pass.
    - **During Forward**: You use `ctx.save_for_backward(inputs, outputs)` to store **Primal Values**. These are the values (like the $x$ in $e^x$) that the derivative formula needs later (actually the input tensors).
    - **The Storage**: This is what "populates" the tape. Without saving these primals, the VJP would not know where on the curve to calculate the slope.
- **The Pointers: `forward` and `backward`**: Each node in the graph holds more than just data; it holds **logic**:
    - The Forward Function: The code that was just executed (e.g. `torch.exp`)
    - The Backward Function (The VJP): A pointer to the specific derivative logic (e.g., `grad_output * result`)

- **The "Tape" is a Chain of `next_functions`**: The "Tape" in PyTorch is actually represented by the `grad_fn` attribute on the tensors.

    - If say $y = a + b$, the tensor $y$ has a `grad_fn` pointing to an `AddBackward` node.
    - That `AddBackward` node has a property called `next_functions`, which points to the `grad_fn` of $a$ and $b$.

**This chain of pointers is the Tape**. When you call `.backward()`, PyTorch simply follows these pointers backward through the graph.

## 5. Conclusive Summary
This blog covers the mathematical definition of Jacobian and how it forms the mathematical basis of Automatic Differenation to its computational implementation using pointers, tapes and topological sorts.

The core takeaway would thus be that: **Automatic Differentiation is the art of implementing the Chain Rule without the baggage of the Jacobian**

- **The Math** gives use the "Master Map" (J) and the operations ($Jv$ and $v^TJ$).
- **The Engineering** gives us the "Tape" and "Pullbacks" to execute those operations efficiently.
- **The Framework (PyTorch/JAX)** manages the "Topological Sort" and "Accumulation" so we can focus on building models instead of manually transposing trillion-element matrices.
___

<!-- React and Babel from CDN -->
<script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>

<!-- Load Custom Components -->
<script type="text/babel" src="/prashant-kanuru-blogs/assets/js/ad-animation.jsx"></script>
<script type="text/babel" src="/prashant-kanuru-blogs/assets/js/pytorch-pointer-animation.jsx"></script>
