---
layout: default
title: "Support Vector Machines"
permalink: /support-vector-machines/
---

I have always been interested in doing a deep dive on **Support Vector Machines** mainly driven by the fact that they use kernels to turn linearly unseparable feature space by using a kernel formed by switching into a higher dimension and the usage of **Kernel** also enthused me to exploring how these can be used to understand the ability of **Universal Approximation Algorithm**, the algorithm that states that (in simple terms), one can approximate any **Arbitrary Continuous Function** by using a neural network, a single layer of neurons, if there is no limit on the number of neurons.

I intend to share my journey as a series of blogs starting with an overview and then moving onto the weeds of the mathematical formulations and possibly gaining some insight into one of the many perspectives on Deep Neural Networks to understand one or more of its building blocks.

## Chapter 1: The overall Perspective

___

### 1. What is SVM?

Support Vector Machine (SVM) falls under the class of supervised machine learning algorithm that can be used for both classification and regression tasks. The core idea is to find the Optimal Separating Hyperplane - a decision boundary that best separates the data points of different classes. The `**"best"**` hyperplane in this case being the one that achieves maximum margin between the nearest training data-points of any class, which are called the support vectors.

### 2. What is Margin (Hard Margin)?

The **hard margin** is the distance between the hyperplane and the nearest data point from either class. Maximizing this distance makes the separation boundary more generalized and thus you can say robust.

A linear hyperplane is defined by the equation:

$$w \cdot x + b = 0$$

where $w$ is the normal vector to the hyperplane, $x$ is a data point and b is the basis (or intercept).

For a dataset with $N$ points $(x_i, y_i)$, where $y_i \in \{-1,1\}$ is the class label, the hard margin constraint is that every point must be correctly classified with a functional margin of at least 1.

$$y_{i}(w \cdot x_{i} + b) \geq 1, \text{for} i = 1,...,N$$

The distance of a point to the hyperplane is $\displaystyle \frac{ \lvert \mathbf{w} \cdot \mathbf{x}_i + b \rvert }{ \lVert \mathbf{w} \rVert }$. The geometric margin $\gamma$ is this distance for the support vectors, and it can be shown that maximizing $\gamma$ is equivalent to minimizing $\lVert \mathbf{w} \rVert^2$:

Primal Form:

$$\min_{\mathbf{w},b}\quad \frac{1}{2}||\mathbf{w}||^2$$

$$\text{subject to}\quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) - 1 \geq 0, \quad \text{for } i = 1,\dots,N$$

### 3. Why Lagrangian Multipliers and Why Use Them in SVM?

The primary optimization problem (minimizing $||\mathbf{w}||^2$ subject to inequality constraints) is a constrained convex optimization problem. The method of Lagrange Multipliers is used to solve such problems.
By introducing non-negative Lagrange multipliers $\alpha_i$ (one for each constraint), a constrained primal problem is turned into an unconstrained **Lagrangian Function** $\mathcal{L}(\mathbf{w},b,\boldsymbol{\alpha})$. This allows taking partial derivatives with respect to the variables and setting them to zero (aka stationarities or the maximas in this case).

Lagrangian Function $\mathcal{L}$:
$$\mathcal{L}(\mathbf{w},b,\boldsymbol{\alpha}) = \frac{1}{2} ||\mathbf{w}||^2 - \sum_{i=1}^{N} \alpha_i\left[y_i(\mathbf{w} \cdot \mathbf{x}_i +b) -1\right]$$
$$\text{where} \alpha_i\geq 0$$

#### Side Note - can be skipped

___

I got into thinking (not sure, if this is the case for others), as in why the constraint is subtracted from the main objective and how will this help minimize the value of $\mathbf{w}$.
The reason for the constraint being subtracted from the objective function is due to the method of **Lagrange Multipliers**, which converts a constrained optimization problem into an unconstrained one.

- The **Original Problem (Primal)**: The original problem is:

$$\min_{\mathbf{w}, b} \quad \frac{1}{2}||\mathbf{w}||^2$$

$$\text{subject to} \quad g_i(\mathbf{w},b) = 1 - y_i(\mathbf{w} \cdot \mathbf{x}_i +b) \leq 0 \quad (\text{Constraint})$$

We want the functional margin term $y_i(\mathbf{w} \cdot \mathbf{x}_i + b)$ to be \geq 1$, so the constraint we want to enforce is $1-y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \leq 0$.

- The Lagrangian Function $\mathcal{L}$: The Lagrangian function is constructed as:

$$\mathcal{L}(\text{variables}) = \text{Objective Function} + \sum(\text{Lagrange Multiplier} \times \text{Constraint})$$

For an inequality constraint $g_i(\mathbf{w}, b) \leq 0$, the standard formulation for the Lagrangian is:
$$\mathcal{L}(\mathbf{w},b, \boldsymbol{\alpha}) =  \frac{1}{2} ||\mathbf{w}||^2 + \sum_{i=1}^N \alpha_i \cdot g_i(\mathbf{w},b) = 1 - y_i(\mathbf{w} \cdot \mathbf{x}_i + b)$$

Substituting $g_i(\mathbf{w},b)  = 1 - y_i(\mathbf{w} \cdot \mathbf{x}_i + b)$:

$$\mathcal{L}(\mathbf{w},b, \boldsymbol{\alpha}) = \frac{1}{2} ||\mathbf{w}||^2 + \sum_{i=1}^N \alpha_i \left[1-y_i(\mathbf{w} \cdot \mathbf{x}_i + b)\right]$$

To match the common SVM formulation (which often groups the terms differently):

$$\mathcal{L}(\mathbf{w},b,\boldsymbol{\alpha}) = \frac{1}{2} ||\mathbf{w}||^2 - \sum_{i=1}^N \alpha_i\left[y_i(\mathbf{w} \cdot \mathbf{x}_i + b) -1 \right]$$

The constraint term is subtracted because the definition of $g_i$ and the sign convention used in the standard SVM formulation lead to a negative sign overall to simplify the subsequent steps (the $\min \max$ problem).

**How this helps Minimize $||\mathbf{w}||^2$**
The Lagrangian does not solve the minimization problem directly; it transforms the original problem ($\min \text{objective subject to constraints}$) into an unconstrained minimax problem:

$$\min_{\mathbf{w},b}\left[\max_{\boldsymbol{\alpha} \geq 0} \mathcal{L}(\mathbf{w},b,\boldsymbol{\alpha})\right]$$
This work because the term $\sum \alpha_i[\dots]$ acts as a penalty that is only deactivated when the constraints are satisfied.

**The Mechanism of the Penalty Term:**
The inner maximization, $\max_{\boldsymbol{\alpha} \geq 0} \mathcal{L}(\mathbf{w},b,\boldsymbol{\alpha})$, forces the constraints to be satisfied:

Constraint Violated: If a data point violates the constraint, $y_i(\mathbf{w} \cdot \mathbf{x}_i +b) -1$ becomes negative. Since $\alpha_i \geq 0$, the entire penalty term $-\alpha_i[\dots]$ becomes positive and grows very large as we maximize $\alpha_i$. This drives $\mathcal{L}$ towards infinity, which means the inner $\max$ step rejects any vector $\mathbf{w}$ that violates the constraint.

Constraint Satisfied: If a data point satisfies the constraint, $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) -1$ is positive or zero. To maximize $\mathcal{L}$, we want to maximize the negative term $\sum \alpha_i \cdot (\text{positive})$. The only way to maximize a function by adding a growing negative term is to set the multiplier $\alpha_i$ to 0. If $\alpha_i = 0$, the penalty term vanishes, and $\mathcal{L}$ reduces to the original objective: $\mathcal{L} = \frac{1}{2} \lvert \mathbf{w} \rvert^2$.

Conclusion: The $\max$ step ensures that only $\mathbf{w}$ and $b$ that satisfy all constraints are allowed to survive and participate in the final minimization step. The outer $\min$ then finds the minimum of $\frac{1}{2} \lvert \mathbf{w} \rvert^2$ among these feasible vectors. This entire process allows us to minimize $\frac{1}{2} \lvert \mathbf{w} \rvert^2$ while respecting the margin requirements.

___

### 4. Dual Problem

___

Solving for the minimum of the Lagrangian function with respect to the primal variables ($\mathbf{w}$ and $b$), leads to the derivation of the Dual Problem. The Dual problem is crucially expresses the entire problem in terms of dot products $\mathbf{x}_i \cdot \mathbf{x}_j$, which enables the use of the Kernel Trick as dot-product instead of adding additional dimensions and this will be explained in detail in the future chapters.

**Derivatives and Constraints (KKT Conditions)**:
We set the partial derivatives of $\mathcal{L}$ to zero;
Derivative w.r.t. $\mathbf{w}$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 0 \quad \implies \quad \mathbf{w} = \sum_{i=1}^N \alpha_i y_i\mathbf{x}_i$$

Derivative w.r.t. $b$:
$$\frac{\partial\mathcal{L}}{\partial b} = 0 \quad\implies \quad \sum_{i=1}^{N} \alpha_i y_i = 0$$

**Dual Objective Function**
The Dual objective is the next step as it is an expression obtained by susbtituting the expression for $\mathbf{w}$ and the constraint on $\sum \alpha_i y_i$ back into $\mathcal{L}$. As stated above the aim is to maximize the value of $\mathbf{w}$ so that the objective function $\frac{1}{2} \quad \lvert \mathbf{w} \rvert^2$.
Completing the substitution of  $\mathbf{w}$ and $b$ into the objective function, would give.

$$\max_{\boldsymbol{\alpha}} \quad \sum_{i=1}^{N} \alpha_i -\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (\mathbf{x}_i \cdot \mathbf{x}_j)$$
$$\text{subject to} \quad \sum_i{i=1}^N \alpha_i y_i = 0 \quad \text{and} \quad \alpha_i \geq 0, \quad \text{for} i=1, \ldots, N$$

___

### 5. Kernel Trick in SVM

___

As was evident in the Dual formulation:
$$\max_{\boldsymbol{\alpha}} \quad \sum_{i=1}^{N} \alpha_i -\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (\mathbf{x}_i \cdot \mathbf{x}_j)$$

it depends only on the dot product of input features $\mathbf{x}_i \cdot \mathbf{x}_j$. The kernel trick is a method that allows SVM to efficiently operate in a high dimensional (or infinite-dimensional) feature space without explicitly calculating the coordinates in that space (I will go through deeper on this in later chapters). So, the mapping into higher dimension (to handle linear non-separability in input feature space) is done by replacing the simple dot product $\mathbf{x}_i \cdot \mathbf{x}_j$. This function implicitly computes the dot product in a higher-dimensional space $\Phi(\mathbf{x})$:
$$K(\mathbf{x}_i,\mathbf{x}_j) = \Phi(\mathbf{x}_i) \cdot \Phi(\mathbf{x}_j)$$

**Kernel Functions**:
By replacing the dot product in the Dual objective, the Dual Problem becomes:
$$\max_{\boldsymbol{\alpha}} \quad \sum_{i=1}^{N} \alpha_i -\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_j y_i y_j K(\mathbf{x}_i, mathbf{x}_j)$$

Examples of Kernel Functions are:

1. Linear Kernel (The original dot product):
$$K(\mathbf{x}_i, \mathbf{x_j}) = \mathbf{x}_i \cdot \mathbf{x}_j$$

2. Polynomial Kernel:
$$K(\mathbf{x}_i,mathbf{x}_j) = (\gamma \mathbf{x}_i \cdot \mathbf{x}_j + r)^d$$

3. Radial Basis Function (RBF) or Gaussian Kernel:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp \left(-\gamma ||\mathbf{x}_i - \mathbf{x}_j||^2 \right)$$,
where $\gamma > 0$ is a hyperparameter.

___

### 6. Karush-Kuhn-Tucker (KKT) Conditions

___

The KKT conditions are the neccessary and sufficient conditions (more detailed explanation on KKT will be followed upon in the later chapters) for the optimal solution $(\mathbfw{w}^*, b^*, \boldsymbol{\alpha}^*)$ of the convex optimization problem (both Primal and Dual). For hard-margin SVM, the KKT complementary condition is:

$$\alpha_i^* \left[y_i(\mathbf{w}^* \cdot \mathbf{x}_i+b^*)-1\right]=0, \quad \text{for} i = 1, \ldots, N$$

This condition tells us that for any data point $x_i$:

- If $\mathbf{x}_i$ is not a support vector (i.e., $y_i(w^*\cdot \mathbf{x}_i+b^*)> 1$, it's outside the margin), then its corresponding multiplier $\alpha_i^*$ must be zero.
- If $\mathbf{x}_i$ is a support vector (i.e., $y_i(\mathbf{w}^* \cdot \mathbf{x}_i+b^*)=1$, it is on the margin), then its $\alpha_i^*$ must be greater than zero.

___

### 7. What is Soft Margin?
___

The **Hard Margin SVM** requires the data to be perfectly linearly separarable, which is rarely the case in real-world data. The **Soft MargM** addresses this by allowing some mis-classification or margin violations, achieving a trade-off between maximizing the margin and minimizing the classification error.

This is achieved by introducing slack variables $\xi_i$ (xi) for each data point, which measure the degree of violation of the margin constraint.

The objective function now includes a penalty term $C \sum \xi_i$:

$$\min_{\mathbf{w},b,\boldsymbol{\xi}} \quad \frac{1}{2}||\mathbf{w}||^2 + C \sum_{i=1}^{N} \xi_i$$

$$\text{subject to} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 -\xi_i, \quad \text{for} i=1, \ldots, N$$

$$\text{and} \quad \xi_i \geq 0$$

$C$ is the regularization parameter (or Box Constraint) which controls the trade-off.
___

### 8. Famous Examples Where SVM is Used?
___

SVMs are highly effective in scenarios where the data is complex or the feature space is high-dimensional:

- **Text and Hypertext Categorization**: Used for classifying documents based on their content.
  **Dataset Schema**:
  - Rows: Individual documents or web-pages
  - Columns (Features): A **Sparse Matrix** of word frequencies. Each column represents a unique word in the vocabulary (often using **TF-IDF** weighting).
  - Shape: Extremely high-dimensional (e.g., 50,000+ features/words) but very sparse (most documents only contain a small fraction of the total vocabulary).

  **Why SVM Fits**:
  - High Dimensionality: SVMs are mathematically robust against the "curse of dimensionality". Because the decision boundary is defined only by support vectors, it does not get "overwhelmed" by the tens of other word features.
  - Linear Separability: Most text classification problems are linearly separable in high-dimensional space, making a linear SVM extremely fast and accurate.
- **Image Classification/Recognition**: Particularly face detection and object recognition.
  **Dataset Schema**:
  - Rows: Inidividual images or cropped patches of images
  - Columns (Features): Normalized pixel intensity values (grayscale/RGB) or extracted feature vectors like HOG(Histogram Oriented Gradients) or SIFT(Scale Invariant Feature Transform).
  - Example: A 64x64 pixel patch results in a 4,096-dimensional input vector.
  **Why SVM Fits**:
  - Global Optimization: Unlike early neural networks, SVMs find the global maximum margin, which is critical for distinguishing subtle differences (e.g., "Is this a face or a random texture")
  - Small Sample Efficiency: In specific recognition tasks where training sets are small SVMs can perform exceptionally well.   
- **Bioinformatics**: Used for protein classification, cancer diagnosis, and gene function prediction.
  **Dataset Schema**:
  - Rows: Patients or biological samples
  - Columns (Features): Expression levels of thousands of genes (Microarray data).
  - The "p >> n" Problem: You might have 20,000 genes (p) but only 100 patients (n).
  **Why SVM fits**:
  - Regularization: SVMs have a built-in regularization parameter (C) that prevents overfitting, which is the biggest risk when you have way more features than samples.
  - Kernel Trick: Biological relationships are rarely linear. SVM Kernels (like RBF) allow the model to capture complex, non-linear interactions between genes without needing to manually define them.
- **Handwriting Recognition.**
  **Dataset Schema**:
  - Rows: Scanned images of handwritten characters (e.g., the MNIST dataset).
  - Columns (Features): 8 bit grayscale values for each pixel.
  - Labels: Discrete classes (0-9, A-Z)
  **Why SVM Fits**:
  - **Multi-Class Versatility**: SVMs though nativley binary can handle multi-class classification by using strategies like One-vs-Rest (OvR) or One-vs-One (OvO). These allow it to handle 10 digits or 26+ letters of the alphabet.
  - **Noise Robustness**: Handwriting is noisy and the "Soft Margin" approach in SVM allows ignoring the outliers to create a generalizable decision boundary for the majority of the data



## Next Chapter: Chapter 2
___

- Why is the distance between Hard Margins 2, i.e. $\frac{2}{||\mathbf{w}||}$
- A lagrangian is basically aligning the gradient of the constraint with that of the object function and why the lagrangian coefficients for non-negative constraints have to be $\geq 0$

## Appendix:

### Note on One-vs-One (OVO)
___

Support Vector Machines (SVMs) are ntaurally **binary classifiers**, meaning they are designed to distinguish between only two classes ( boolean).
To handle **Multi-Class Classification**, One-vs-One (OVO) strategy.

#### How OvO Works
In the OvO approach, a multi-class problem into a series of **binary pairings**. A separate classifier is trained for every possible pair of classes.

**1. The Formula**
In a generalized sense, for $K$ classes, the number of individual binary classifiers needed to train is calculated using the formula:

$$\binom{K}{2} = \frac{K(K-1)}{2}$$
Example with 4 classes (A, B, C, D):

Number of classifiers needed: $\frac{4(4-1)}{2} = 6$

A vs B

A vs C

A vs D

B vs C

B vs D

C vs D
___

#### How the Winner is Chosen (Max-Wins Voting)

When a new unseen data point is to be classified:

1. **Run all classifiers**: The data point is passed through all 6 classifiers
2. **Collect Votes**: Each classifier predicts one of its two classes. For example, the "A vs. B" classifier might vote for "A".
3. **Final Decision**: The class that receives the **highest number of votes** across all classifiers is the final prediction.
___

#### OvO vs. OvR (One-vs-Rest)
The other common strategy is **One-vs-Rest(OVR)**, in which one classifier is trained per class (e.g., "Apple vs. Everything else"). Here is how they compare:

|Feature |One-vs-One(OvO) |One-vs-Rest(OvR) |
|--- |--- |--- |
|Number of models |Much Higher (K(K-1)/2) |Lower(K) |
|Training Speed |**Fast** |**Slow** |
|Memory Usage|High (storing many models).|Low (storing few models) |
|Preference |Preferred for SVMs and kernel-based methods |Preferred for **Logistic Regression** |

___

#### Why is OvO preferred for SVMs?

SVM training complexity is more than linear relative to the number of samples.
- In **OvR**, each classifier is trained on the **entire dataset**, which can be very slow if the dataset is large.
- In **OvO**, each classifier is trained only on data from **two classes**. Since the training set for each binary sub-problem is much smaller, the individual SVMs train much faster, often offsetting the "burden" of having to train many models.

