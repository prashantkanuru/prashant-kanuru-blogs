I have always been interested in doing a deep dive by the fact that they use kernels to turn linearly unseparable feature space by virtue of a kernel formed by switching into a higher dimension and always felt like exploring how these can be used to understand the ability of **Universal Approximation Algorithm**.

I intend to share my journey as a series of blogs starting with an overview and then movin onto the weeds of the mathematical formulations and possibly gaining some insight into one of the many perspectives of Deep Neural Networks can be understood.

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

The distance of a point to the hyperplane is $\frac{|\mathbf{w} \cdot \mathbf{x}_i + b|}{||\mathbf{w}||}$. The geometric margin $\gamma$ is this distance for the support vectors, and it can be shown that maximizing $\gamma$ is this distance for the support vectors, and it can be shown that maximizing $\gamma$ is equivalent to minimizing $||\mathbf{w}||^2$:

Primal Form:

$$\min_{\mathbf{w},b}\quad \frac{1}{2}||\mathbf{w}||^2$$

$$\text{subject to}\quad y_i(\mathbf{w} \cdot \mathbf{x_i} +b) -1 \geq 0, \quad \text{for}i = 1,\|dots, N$$

### 3. Why Lagrangian Multipliers and Why Use Them in SVM?
The primary optimization problem (minimizing $||\mathbf{w}||^2$ subject to inequality constraints) is a constrained convex optimization problem. The method of Lagrange Multipliers is used to solve such problems.
By introducing non-negative Lagrange multipliers $\alpha_i$ (one for each constraint), a constrained primal problem is turned into an unconstrained **Lagrangian Function** $\mathcal{L}(\mathbf{w},b,\boldsymbol{\alpha})$. This allows taking partial derivatives with respect to the variables and setting them to zero (aka stationarities or the maximas in this case).

Lagrangian Function $\mathcal{L}$:
$$\mathcal{L}(\mathbf{w},b,\boldsymbol{\alpha}) = \frac{1}{2} ||\mathbf{w}||^2 - \sum_{i=1}^{N} \alpha_i\left[y_i(\mathbf{w} \cdot \mathbf{x}_i +b) -1\right]$$
$$\text{where} \alpha_i\geq 0$$





### Next Chapter: Chapter 2:
- Why is the distance between Hard Margins 2, i.e. $\frac{2}{||\mathbf{w}||}$
- A lagrangian is basically aligning the gradient of the constraint with that of the object function and why the lagrangian coefficients for non-negative constraints have to be $\geq 0$
