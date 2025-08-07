import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
from copy import deepcopy

# -- 1. Setup: Define Data and Model --

# Use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_and_labels(n_samples):
    """Generates samples from a mixture of three Gaussians using only PyTorch."""
    mus = torch.tensor([-4.0, 0.0, 4.0])
    std = 0.1

    labels = torch.randint(0, 3, (n_samples,))
    means = mus[labels]
    data = torch.randn(n_samples) * std + means

    return data.unsqueeze(1), labels


# Model: MLP with a learned embedding for unconditional generation
class ConditionalFlowMLP(nn.Module):
    def __init__(self, input_dim=1, context_dim=3, hidden_dim=128):
        super().__init__()
        self.null_embedding = nn.Parameter(torch.randn(1, context_dim))

        self.net = nn.Sequential(
            nn.Linear(
                input_dim + context_dim + 2, hidden_dim
            ),  # +2 for time and integrator flag
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, t, context, integrate=False):
        if integrate:
            net_input = torch.cat(
                [x, t, context, torch.ones_like(t) * integrate], dim=1
            )
        else:
            net_input = torch.cat([x, t, context, torch.zeros_like(t)], dim=1)
        return self.net(net_input)


# -- 2. Training with Classifier-Free Guidance --


def train_flow_matcher_with_cfg(
    # Hyperparameters
    n_samples=4000,
    batch_size=32,
    epochs=100,
    lr=1e-3,
    n_classes=3,
    p_uncond=0.1,
):
    """trains the conditional flow model with support for CFG."""

    model = ConditionalFlowMLP(context_dim=n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    data, labels = get_data_and_labels(n_samples)

    print("Starting Training with Classifier-Free Guidance...")
    for epoch in range(epochs):
        permutation = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            indices = permutation[i : i + batch_size]
            x1, batch_labels = data[indices].to(device), labels[indices].to(device)

            c = nn.functional.one_hot(batch_labels, num_classes=n_classes).float()
            uncond_mask = torch.rand(c.size(0), device=device) < p_uncond
            c[uncond_mask] = model.null_embedding

            x0 = torch.randn_like(x1)
            t = torch.rand(x1.size(0), 1, device=device)
            # r = torch.rand(x1.size(0), 1, device=device)
            # t = 1 full data, t = 0 full prior
            xt = (1 - t) * x0 + t * x1
            target_velocity = x1 - x0

            predicted_velocity = model(xt, t, c)

            loss = loss_fn(predicted_velocity, target_velocity)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    print("Training finished.")
    return model


def fine_tune_flow_matcher_with_cfg_student_teacher(
    model=None,
    # Hyperparameters
    n_samples=4000,
    batch_size=32,
    epochs=100,
    lr=1e-3,
    n_classes=3,
    p_uncond=0.1,
):
    """trains the conditional integrator model from the conditional flow model with support for CFG"""

    if model is None:
        model = ConditionalFlowMLP(context_dim=n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    data, labels = get_data_and_labels(n_samples)
    teacher = deepcopy(model)

    print("Starting Training with Classifier-Free Guidance...")
    for epoch in range(epochs):
        permutation = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            indices = permutation[i : i + batch_size]
            x1, batch_labels = data[indices].to(device), labels[indices].to(device)

            c = nn.functional.one_hot(batch_labels, num_classes=n_classes).float()
            uncond_mask = torch.rand(c.size(0), device=device) < p_uncond
            c[uncond_mask] = model.null_embedding

            x0 = torch.randn_like(x1)
            t = torch.rand(x1.size(0), 1, device=device)
            # t = 0 full prior, t = 1 full data
            # x0 = prior, x1 = data
            xt = (1 - t) * x0 + t * x1
            target_velocity = x1 - x0

            predicted_velocity = model(xt, t, c)

            # second order, ODE (heun) is being integrated here as an effective target 
            # when the integrate flag is enabled the model switched mode into an integrator model
            # technically we could use better integrator here like rk45 but heun is the cheapest
            r = torch.rand(x1.size(0), 1, device=device)
            r = torch.max(r, t)
            # "dt" = vector spanning from t to r
            dt = r - t
            # now we step for 1 ode step to obtain rt point and use it as input for the second order
            rt = xt + predicted_velocity * dt
            with torch.no_grad():
                w = torch.rand(x1.size(0), 1, device=device)
                predicted_velocity_second_order = teacher(rt, r, c)
                target_velocity = (
                    predicted_velocity_second_order + predicted_velocity
                ) / 2

            loss = loss_fn(predicted_velocity, target_velocity)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    print("Training finished.")
    return model


def train_flow_matcher_with_cfg_and_self_integrator(
    model=None,
    # Hyperparameters
    n_samples=4000,
    batch_size=32,
    epochs=100,
    lr=1e-3,
    n_classes=3,
    p_uncond=0.1,
    p_integrator=0.5,
):
    """trains the conditional flow model with built in integrator mode with support for CFG."""

    if model is None:
        model = ConditionalFlowMLP(context_dim=n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    data, labels = get_data_and_labels(n_samples)

    print("Starting Training with Classifier-Free Guidance...")
    for epoch in range(epochs):
        permutation = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            indices = permutation[i : i + batch_size]
            x1, batch_labels = data[indices].to(device), labels[indices].to(device)

            c = nn.functional.one_hot(batch_labels, num_classes=n_classes).float()
            uncond_mask = torch.rand(c.size(0), device=device) < p_uncond
            c[uncond_mask] = model.null_embedding

            x0 = torch.randn_like(x1)
            t = torch.rand(x1.size(0), 1, device=device)
            # t = 0 full prior, t = 1 full data
            # x0 = prior, x1 = data
            xt = (1 - t) * x0 + t * x1
            target_velocity = x1 - x0

            integrate = False
            if torch.rand(1, device=device) < p_integrator:
                integrate = True
            predicted_velocity = model(xt, t, c, integrate)

            if integrate:
                # second order, ODE (heun) is being integrated here as an effective target 
                # when the integrate flag is enabled the model switched mode into an integrator model
                # technically we could use better integrator here like rk45 but heun is the cheapest
                r = torch.rand(x1.size(0), 1, device=device)
                r = torch.max(r, t)
                # "dt" = vector spanning from t to r
                dt = r - t
                # now we step for 1 ode step to obtain rt point and use it as input for the second order
                rt = xt + predicted_velocity * dt
                with torch.no_grad():
                    w = torch.rand(x1.size(0), 1, device=device)
                    predicted_velocity_second_order = model(rt, r, c, False)
                    target_velocity = (
                        predicted_velocity_second_order + predicted_velocity
                    ) / 2

            loss = loss_fn(predicted_velocity, target_velocity)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    print("Training finished.")
    return model


# -- 3. Inference and Visualization --


def generate_trajectories(
    model, n_samples, n_steps, context, cfg_scale, integrate=False
):
    """Core function to generate trajectories with CFG."""
    model.eval()
    c_cond = context.repeat(n_samples, 1)
    c_uncond = model.null_embedding.repeat(n_samples, 1)

    x0 = torch.randn(n_samples, 1, device=device)
    trajectory = torch.zeros(n_steps + 1, n_samples, device=device)
    trajectory[0] = x0.squeeze()
    x = x0
    dt = 1.0 / n_steps

    with torch.no_grad():
        for i in range(n_steps):
            t_val = i * dt
            t = torch.ones(n_samples, 1, device=device) * t_val
            v_cond = model(x, t, c_cond, integrate)
            v_uncond = model(x, t, c_uncond, integrate)
            velocity = v_uncond + cfg_scale * (v_cond - v_uncond)
            x = x + velocity * dt
            trajectory[i + 1] = x.squeeze()

    return trajectory.cpu().numpy().T


def _plot_ghost_distributions(ax, ghost_data, y_range=(-7, 7), bins=200):
    """Helper function to plot the background distributions."""
    # Plot ghost prior distribution at t=0
    prior_noise = np.random.randn(2000)
    hist_prior, bin_edges_prior = np.histogram(
        prior_noise, bins=bins, range=y_range, density=True
    )
    centers_prior = (bin_edges_prior[:-1] + bin_edges_prior[1:]) / 2
    # Plot as horizontal bars pointing left from t=0
    ax.barh(
        centers_prior,
        -hist_prior,
        height=(y_range[1] - y_range[0]) / bins,
        left=0,
        color="gray",
        alpha=0.25,
        zorder=0,
        label="Prior & Target Distributions",
    )

    # Plot ghost target distribution at t=1
    hist_target, bin_edges_target = np.histogram(
        ghost_data.flatten(), bins=bins, range=y_range, density=True
    )
    centers_target = (bin_edges_target[:-1] + bin_edges_target[1:]) / 2
    # Plot as horizontal bars pointing right from t=1
    ax.barh(
        centers_target,
        hist_target,
        height=(y_range[1] - y_range[0]) / bins,
        left=1,
        color="gray",
        alpha=0.25,
        zorder=0,
    )


def plot_trajectories(trajectory, title, file_path, ghost_data):
    """Plots and saves a static image of the trajectories with ghost distributions."""
    n_samples, n_steps = trajectory.shape
    n_steps -= 1
    y_range = (-7, 7)

    fig, ax = plt.subplots(figsize=(10, 6))
    time_axis = np.linspace(0, 1, n_steps + 1)

    _plot_ghost_distributions(ax, ghost_data, y_range=y_range)

    segments = []
    for i in range(n_samples):
        points = np.array([time_axis, trajectory[i, :]]).T.reshape(-1, 1, 2)
        segments.extend(np.concatenate([points[:-1], points[1:]], axis=1))

    line_collection = LineCollection(
        segments, color="royalblue", alpha=0.2, linewidths=0.5
    )
    ax.add_collection(line_collection)
    ax.scatter(
        np.zeros(n_samples),
        trajectory[:, 0],
        color="darkorange",
        s=10,
        zorder=3,
        label="Initial Noise (t=0.0)",
    )
    ax.scatter(
        np.ones(n_samples),
        trajectory[:, -1],
        color="navy",
        s=10,
        zorder=3,
        label="Generated Data (t=1.0)",
    )

    ax.set_xlim(-0.5, 1.5)  # Widen x-axis for ghost plots
    ax.set_ylim(y_range)
    ax.set_title(title)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Value (x)")
    # Manually handle legend to avoid duplicate ghost labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(file_path, dpi=150)
    print(f"Static image saved to {file_path}")
    plt.close()


if __name__ == "__main__":
    BASE_EPOCHS = 3000
    P_INTEGRATE = 0.5
    torch.manual_seed(0)
    np.random.seed(0)
    output_dir = f"simple_integrator"
    os.makedirs(output_dir, exist_ok=True)

    # --- Get Full Data Distribution for Ghost Plots ---
    full_data_dist, _ = get_data_and_labels(50000)

    trained_model = train_flow_matcher_with_cfg_and_self_integrator(
        epochs=BASE_EPOCHS, p_integrator=P_INTEGRATE
    )
    torch.save(trained_model, f"{output_dir}/flow_integrator.pth")

    # --- Set Inference Parameters ---
    n_gen_samples = 500
    n_gen_steps = 10
    cfg_scale = 1
    torch.manual_seed(0)
    np.random.seed(0)
    # --- 1. Conditional Generation for each Label ---
    for label in range(3):
        print(f"\n--- Generating for Label: {label} (CFG Scale: {cfg_scale}) ---")
        context_vec = (
            nn.functional.one_hot(torch.tensor([label]), num_classes=3)
            .float()
            .to(device)
        )
        traj = generate_trajectories(
            trained_model, n_gen_samples, n_gen_steps, context_vec, cfg_scale
        )

        title_str = (
            f"Conditional Flow Trajectories | Label: {label}, CFG Scale: {cfg_scale}"
        )
        img_path = os.path.join(
            output_dir, f"static_label_{label}_cfg_{cfg_scale}_instant.png"
        )
        plot_trajectories(traj, title_str, img_path, ghost_data=full_data_dist)

    # --- 2. Unconditional Generation ---
    print(f"\n--- Generating Unconditionally (CFG Scale: 0.0) ---")
    uncond_context = torch.zeros(1, 3, device=device)
    traj_uncond = generate_trajectories(
        trained_model, n_gen_samples, n_gen_steps, uncond_context, cfg_scale=0.0
    )

    title_uncond = "Unconditional Flow Trajectories"
    img_path_uncond = os.path.join(output_dir, "static_unconditional_instant.png")
    plot_trajectories(
        traj_uncond, title_uncond, img_path_uncond, ghost_data=full_data_dist
    )

    torch.manual_seed(0)
    np.random.seed(0)
    # --- 1. Conditional Generation for each Label ---
    for label in range(3):
        print(f"\n--- Generating for Label: {label} (CFG Scale: {cfg_scale}) ---")
        context_vec = (
            nn.functional.one_hot(torch.tensor([label]), num_classes=3)
            .float()
            .to(device)
        )
        traj = generate_trajectories(
            trained_model,
            n_gen_samples,
            n_gen_steps,
            context_vec,
            cfg_scale,
            integrate=True,
        )

        title_str = (
            f"Conditional Flow Trajectories | Label: {label}, CFG Scale: {cfg_scale}"
        )
        img_path = os.path.join(
            output_dir, f"static_label_{label}_cfg_{cfg_scale}_integrated_lerp.png"
        )
        plot_trajectories(traj, title_str, img_path, ghost_data=full_data_dist)

    # --- 2. Unconditional Generation ---
    print(f"\n--- Generating Unconditionally (CFG Scale: 0.0) ---")
    uncond_context = torch.zeros(1, 3, device=device)
    traj_uncond = generate_trajectories(
        trained_model,
        n_gen_samples,
        n_gen_steps,
        uncond_context,
        cfg_scale=0.0,
        integrate=True,
    )

    title_uncond = "Unconditional Flow Trajectories"
    img_path_uncond = os.path.join(
        output_dir, "static_unconditional_integrated_lerp.png"
    )
    plot_trajectories(
        traj_uncond, title_uncond, img_path_uncond, ghost_data=full_data_dist
    )
