### Incorporating an ODE Solver as an Effective Target in Conditional Flow Matching

This repository presents a simple approach to training conditional flow matching (CFM) models by incorporating an Ordinary Differential Equation (ODE) solver as an effective target during training. This method allows the model to learn not only the instantaneous vector field but also an integrated vector field, leading to more robust and efficient generation.

The training is conditioned to enable the model to generate either the ordinary vector field induced by the standard CFM loss or an integrated vector field. This is achieved by introducing a flag during training that signals to the model which target to learn.

---

### Background: Conditional Flow Matching

Conditional Flow Matching (CFM) is a powerful technique for training generative models. The core idea is to learn a vector field that transports samples from a simple prior distribution (e.g., a Gaussian) to a complex data distribution. The standard CFM loss is a regression-based objective that minimizes the difference between the model's predicted vector field and a target vector field.

The standard CFM loss can be expressed as:

$L_{CFM}(\theta) = {t \sim U(0,1), x_1 \sim p_{data}(x), x_0 \sim p_0(x)} [\| v_\theta(x_t, t) - (x_1 - x_0) \|^2]$

where:
- $x_0$ is a sample from the prior distribution.
- $x_1$ is a sample from the data distribution.
- $t$ is a time step sampled uniformly from.
- $x_t = (1-t)x_0 + t x_1$ is a point on the path between $x_0$ and $x_1$.
- $v_\theta(x_t, t)$ is the velocity predicted by the model at point $x_t$ and time $t$.
- $x_1 - x_0$ is the constant velocity of the straight path from $x_0$ to $x_1$.

---

### Integrating an ODE Solver into the Training Loop

In this work, we extend the standard CFM framework by introducing a second-order ODE solver into the training process. The key innovation is to use the output of an ODE solver as an *effective target* for the model's predicted velocity. This is controlled by an `integrate` flag. When this flag is active, instead of regressing towards the simple `x1 - x0` target, the model is trained to predict a velocity that aligns with a more accurately integrated step.

#### Generalizing the ODE Solver: From Heun to RK45

The provided code implements this concept using **Heun's method**, a simple and computationally inexpensive second-order predictor-corrector solver. The process for this integrated target is as follows:

1.  **Prediction (Euler Step):** A preliminary future point, `rt`, is estimated using a standard Euler step from the current point `xt` with the model's predicted velocity.
2.  **Correction:** The model then predicts the velocity at this new point `rt`.
3.  **Averaging:** The final target velocity is the average of the initial predicted velocity and the velocity at the new point, which is the core of Heun's method.

This can be formulated as:

$v_{target, Heun} = \frac{v_\theta(x_t, t, c) + v_\theta(x_t + v_\theta(x_t, t, c) \cdot (r-t), r, c)}{2}$

where:
- $r$ is a future time step.
- $c$ is the conditioning vector.

However, this framework is highly flexible and not limited to Heun's method. The core principle of using a more sophisticated integration step as a training target can be extended to higher-order and adaptive step-size solvers, such as the popular **Runge-Kutta 45 (RK45)** method.

Using a solver like RK45 would involve calculating a weighted average of velocities from several intermediate steps to compute the integrated target. While this would increase the computational cost per training step due to multiple model evaluations, it offers significant potential advantages:
*   **More Accurate Targets:** RK45 provides a much more accurate approximation of the true integral of the vector field. Training on these superior targets could lead to a model that learns a more robust and globally consistent vector field.
*   **Improved Generation Efficiency:** A model trained with RK45-based targets might be capable of taking larger, more stable steps during inference, potentially reducing the number of function evaluations (NFE) needed to generate high-quality samples.

By stochastically switching between the standard CFM target and this integrated target (controlled by `p_integrator`), the model learns to be a versatile vector field generator. It can produce the instantaneous field required for standard ODE sampling and a more "averaged" or "integrated" field that can potentially allow for larger, more stable generation steps.

---

### Conditional Generation and Classifier-Free Guidance

The model is trained conditionally, meaning it can generate samples belonging to specific classes. This is achieved by providing a one-hot encoded class label as conditioning information to the model.

Furthermore, the implementation utilizes Classifier-Free Guidance (CFG). CFG is a technique that allows for controlling the trade-off between sample diversity and fidelity to the conditioning signal during inference, without needing an external classifier. This is implemented by occasionally dropping the conditioning information during training and using a learned "null" embedding instead. At inference time, the final velocity is a weighted combination of the conditionally and unconditionally predicted velocities.

---

### How to Cite

If you use this work in your research, please cite this repository as follows:

```
@misc{rock2025integrate,
  author = {Lodestone Rock},
  title = {Integrate the Integrator: Incorporating an ODE Solver as an Effective Target in Conditional Flow Matching},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lodestone-rock/integrate-the-integrator}},
}
```
