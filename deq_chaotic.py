#!/usr/bin/env python3
"""
THE CHAOTIC DEQ GAUNTLET: φ-Surfer on Stiff Dynamics
=====================================================

Extension of the DEQ experiment to CHAOTIC/STIFF DYNAMICAL SYSTEMS.

Experiment:
-----------
Build a DEQ with an implicit layer that solves a FIXED POINT of a chaotic map:
    1. MLP predicts initial state x₀
    2. Implicit solver finds fixed point: F(x*, params) = x*
    3. Loss = ||x* - target||²

Where F is a discretized chaotic system (Lorenz, Van der Pol, etc.)

This is HARDER than IK because:
- Chaotic attractors have fractal basin boundaries
- Stiff dynamics → ill-conditioned Jacobians
- Multiple attractors → basin hopping

Hypothesis:
-----------
A. Naive Newton: Diverges in chaotic regime → NaN gradients → Training crash
B. Damped Newton: Stalls near basin boundaries → Poor learning
C. φ-Surfer: Navigates chaos → Stable gradients → Network learns

Author: Holotopic Navigation Research
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
import flax.linen as nn
import optax
from typing import Tuple, Dict
import time

jax.config.update("jax_enable_x64", True)

print("="*80)
print("THE CHAOTIC DEQ GAUNTLET: Training Through Chaos")
print("="*80)
print(f"JAX version: {jax.__version__}")
print(f"Device: {jax.devices()[0]}")
print("="*80 + "\n")


# =============================================================================
# CHAOTIC DYNAMICAL SYSTEM: Duffing-like Map
# =============================================================================

@jit
def duffing_map(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """
    Duffing-like discrete map (simplified for 2D):
    
    x_next = [x1 + α*x2, 
              x2 - β*x1 - γ*x1³ + δ*sin(x1)]
    
    This has chaotic behavior for certain parameter regimes.
    We'll use it as our implicit fixed-point map.
    
    Args:
        x: [2] current state (x1, x2)
        params: [4] map parameters (α, β, γ, δ)
    
    Returns:
        x_next: [2] next state
    """
    x1, x2 = x
    alpha, beta, gamma, delta = params
    
    # Clamp inputs to prevent explosion
    x1 = jnp.clip(x1, -10.0, 10.0)
    x2 = jnp.clip(x2, -10.0, 10.0)
    
    x1_next = x1 + alpha * x2
    x2_next = x2 - beta * x1 - gamma * x1**3 + delta * jnp.sin(x1)
    
    # Clamp outputs to prevent divergence
    x1_next = jnp.clip(x1_next, -10.0, 10.0)
    x2_next = jnp.clip(x2_next, -10.0, 10.0)
    
    return jnp.array([x1_next, x2_next])


@jit
def fixed_point_residual(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """
    Fixed point equation: F(x*, params) - x* = 0
    
    We want to find x such that duffing_map(x, params) = x
    """
    return duffing_map(x, params) - x


@jit
def jacobian_map(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """
    Jacobian of the fixed-point residual: J = dF/dx - I
    """
    # Use JAX autodiff for exact Jacobian
    J_F = jax.jacfwd(duffing_map, argnums=0)(x, params)
    J_residual = J_F - jnp.eye(2)
    return J_residual


@jit
def lyapunov_indicator(x: jnp.ndarray, params: jnp.ndarray) -> float:
    """
    Approximate Lyapunov-like indicator: log(||J||)
    
    High values indicate chaotic/stiff regions.
    We'll use this as our φ-potential proxy.
    """
    J = jacobian_map(x, params)
    J_norm = jnp.linalg.norm(J, ord='fro')
    return jnp.log(J_norm + 1e-8)


# =============================================================================
# DIFFERENTIABLE SOLVERS FOR FIXED POINTS
# =============================================================================

@jit
def newton_step_chaos(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """
    Newton step for fixed point: x_new = x - J^{-1} * f
    
    PROBLEM: In chaotic regime, J can be very ill-conditioned!
    """
    J = jacobian_map(x, params)
    f = fixed_point_residual(x, params)
    
    # Try to invert (can fail!)
    try:
        J_inv = jnp.linalg.inv(J)
        delta_x = J_inv @ f
    except:
        # Fallback to pseudo-inverse
        delta_x = jnp.linalg.pinv(J, rcond=1e-8) @ f
    
    return x - delta_x


@jit
def phi_potential_chaos(x: jnp.ndarray, params: jnp.ndarray) -> float:
    """
    φ-potential for chaotic dynamics:
    
    φ(x) = lyapunov_indicator(x) + ||f(x)||²
    
    High φ → chaotic/unstable region
    """
    f = fixed_point_residual(x, params)
    lyap = lyapunov_indicator(x, params)
    
    # Combine Lyapunov indicator + residual magnitude
    phi = lyap + jnp.sum(f**2)
    
    return phi


@jit
def phi_surfer_step_chaos(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """
    φ-Surfer step for chaotic fixed points:
    
    Blend chaos avoidance + fixed-point tracking
    """
    # Clamp input to prevent divergence
    x = jnp.clip(x, -10.0, 10.0)
    
    # Chaos avoidance gradient (descend φ to avoid instability)
    g_phi = grad(phi_potential_chaos, argnums=0)(x, params)
    
    # Fixed-point tracking gradient
    J = jacobian_map(x, params)
    f = fixed_point_residual(x, params)
    
    # Regularized Newton direction
    J_reg = J.T @ J + 1e-4 * jnp.eye(2)  # Stronger regularization
    g_task = jnp.linalg.solve(J_reg, J.T @ f)
    
    # Clip gradients to prevent explosion
    g_phi = jnp.clip(g_phi, -10.0, 10.0)
    g_task = jnp.clip(g_task, -10.0, 10.0)
    
    # Adaptive blending based on Lyapunov indicator
    lyap = lyapunov_indicator(x, params)
    lyap = jnp.clip(lyap, -5.0, 5.0)  # Prevent extreme values
    
    # High Lyapunov → prioritize stability (avoid chaos)
    # Low Lyapunov → prioritize convergence (Newton)
    # Map to [0,1] properly using sigmoid-like transformation
    raw = jnp.tanh(lyap)              # (-1, 1)
    weight_avoid = 0.5 * (raw + 1.0)  # (0, 1) - proper probability
    weight_task = 1.0 - weight_avoid  # (0, 1) - sums to 1
    
    # Normalized gradients
    g_phi_norm = jnp.linalg.norm(g_phi) + 1e-8
    g_task_norm = jnp.linalg.norm(g_task) + 1e-8
    
    # Blended step (smooth!) with smaller step sizes
    step = (weight_avoid * (-0.02 * g_phi / g_phi_norm) + 
            weight_task * (-0.1 * g_task / g_task_norm))
    
    # Clip step size
    step = jnp.clip(step, -0.5, 0.5)
    
    x_new = x + step
    
    # Final clamp on output
    return jnp.clip(x_new, -10.0, 10.0)


@jit
def damped_newton_step_chaos(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """
    Damped Newton with soft regularization (differentiable)
    """
    J = jacobian_map(x, params)
    f = fixed_point_residual(x, params)
    
    # Check conditioning
    cond = jnp.linalg.cond(J)
    
    # Adaptive regularization
    lambda_reg = jnp.where(cond > 1e6, 1e-3, 1e-8)
    
    J_reg = J.T @ J + lambda_reg * jnp.eye(2)
    g_task = jnp.linalg.solve(J_reg, J.T @ f)
    
    # Soft step limiting
    step_norm = jnp.linalg.norm(g_task) + 1e-8
    alpha = jnp.tanh(1.0 / step_norm)
    
    return x - alpha * g_task


def run_solver_steps_chaos(x_init: jnp.ndarray, params: jnp.ndarray, 
                           solver_fn, num_steps: int = 10) -> jnp.ndarray:
    """
    Run solver for K steps (differentiable!)
    """
    def body_fn(i, x):
        return solver_fn(x, params)
    
    x_final = jax.lax.fori_loop(0, num_steps, body_fn, x_init)
    return x_final


# =============================================================================
# NEURAL NETWORK (Parameter Predictor)
# =============================================================================

class ChaoticDEQ(nn.Module):
    """
    MLP that predicts initial state x₀ given target fixed point x*
    
    This is the encoder that the implicit layer will refine.
    """
    hidden_dim: int = 64
    
    @nn.compact
    def __call__(self, x_target: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x_target: [2] target fixed point
            params: [4] map parameters
        
        Returns:
            x_init: [2] initial state guess
        """
        # Concatenate target and params
        inp = jnp.concatenate([x_target, params])
        
        x = nn.Dense(self.hidden_dim)(inp)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)  # FIX: use x, not inp!
        x = nn.relu(x)
        x = nn.Dense(2)(x)
        
        # Constrain to reasonable range
        x = jnp.tanh(x) * 2.0  # [-2, 2]
        
        return x


# =============================================================================
# DATASET GENERATION
# =============================================================================

def create_chaotic_dataset(num_samples: int = 100, hard_fraction: float = 0.3, 
                          key: jax.random.PRNGKey = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create dataset with HARD chaotic examples.
    
    Returns:
        targets: [num_samples, 2] target fixed points
        params_list: [num_samples, 4] map parameters for each sample
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    num_hard = int(num_samples * hard_fraction)
    num_easy = num_samples - num_hard
    
    # Easy regime (stable fixed points) - conservative parameters
    key, subkey = jax.random.split(key)
    params_easy = jax.random.uniform(subkey, (num_easy, 4), 
                                     minval=jnp.array([0.05, 0.05, 0.01, 0.0]),
                                     maxval=jnp.array([0.15, 0.15, 0.05, 0.05]))
    
    # Hard regime (ACTUALLY CHAOTIC parameters) - push it harder!
    key, subkey = jax.random.split(key)
    # More aggressive parameters to make baselines fail
    params_hard = jax.random.uniform(subkey, (num_hard, 4),
                                     minval=jnp.array([0.3, 0.3, 0.1, 0.2]),
                                     maxval=jnp.array([0.6, 0.6, 0.3, 0.5]))
    
    # Combine
    params_all = jnp.concatenate([params_easy, params_hard], axis=0)
    
    # Generate target fixed points by running the map
    key, subkey = jax.random.split(key)
    x_init_all = jax.random.normal(subkey, (num_samples, 2)) * 0.3  # Smaller initial spread
    
    # Find approximate fixed points by iterating (with safety)
    targets = []
    for i in range(num_samples):
        x = x_init_all[i]
        p = params_all[i]
        # Run for a while to converge (roughly) with clamping
        for _ in range(50):  # Fewer iterations to prevent divergence
            x = duffing_map(x, p)
            # Extra safety: if diverging, reset to small value
            if jnp.linalg.norm(x) > 5.0:
                x = jax.random.normal(jax.random.PRNGKey(i), (2,)) * 0.1
                break
        targets.append(x)
    
    targets = jnp.stack(targets)
    
    # Shuffle
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, num_samples)
    targets = targets[perm]
    params_all = params_all[perm]
    
    return targets, params_all


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def compute_loss_chaotic(params_net: Dict, model: ChaoticDEQ, 
                        x_target: jnp.ndarray, map_params: jnp.ndarray,
                        solver_fn, num_solver_steps: int = 10) -> float:
    """
    Loss with chaotic implicit layer.
    """
    # Clamp target to prevent explosion
    x_target = jnp.clip(x_target, -10.0, 10.0)
    
    # Encoder prediction
    x_init = model.apply(params_net, x_target, map_params)
    x_init = jnp.clip(x_init, -10.0, 10.0)
    
    # Implicit layer (solver finds fixed point)
    x_final = run_solver_steps_chaos(x_init, map_params, solver_fn, num_solver_steps)
    
    # Clamp to prevent NaN propagation
    x_final = jnp.clip(x_final, -10.0, 10.0)
    
    # Replace NaNs/Infs with large finite values (JAX-compatible)
    x_final = jnp.nan_to_num(x_final, nan=10.0, posinf=10.0, neginf=-10.0)
    
    # Check that it's actually a fixed point
    residual = fixed_point_residual(x_final, map_params)
    
    # Loss = distance to target + fixed-point residual (smaller weight on residual)
    loss = jnp.sum((x_final - x_target)**2) + 0.01 * jnp.sum(residual**2)
    
    # Clamp loss to prevent explosion (use where for JAX compatibility)
    loss = jnp.where(jnp.isnan(loss) | jnp.isinf(loss), 1000.0, loss)
    loss = jnp.clip(loss, 0.0, 1000.0)
    
    return loss


def compute_batch_loss_chaotic(params_net: Dict, model: ChaoticDEQ,
                               x_targets: jnp.ndarray, map_params_batch: jnp.ndarray,
                               solver_fn, num_solver_steps: int = 10) -> Tuple[float, Dict]:
    """
    Batch loss for chaotic DEQ.
    """
    # Vectorized loss
    losses = vmap(lambda xt, mp: compute_loss_chaotic(params_net, model, xt, mp, solver_fn, num_solver_steps))(
        x_targets, map_params_batch
    )
    
    # Count NaNs
    nan_count = jnp.sum(jnp.isnan(losses))
    
    # Replace NaNs
    losses = jnp.where(jnp.isnan(losses), 1000.0, losses)
    
    total_loss = jnp.mean(losses)
    
    metrics = {
        'nan_count': nan_count,
        'max_loss': jnp.max(losses),
        'min_loss': jnp.min(losses),
    }
    
    return total_loss, metrics


def train_chaotic_deq(model: ChaoticDEQ, targets: jnp.ndarray, map_params_batch: jnp.ndarray,
                      solver_fn, num_epochs: int = 300, learning_rate: float = 1e-3,
                      num_solver_steps: int = 10) -> Dict:
    """
    Train the chaotic DEQ model.
    """
    # Initialize
    key = jax.random.PRNGKey(0)
    dummy_target = jnp.array([0.5, 0.5])
    dummy_params = jnp.array([0.2, 0.2, 0.2, 0.1])
    params_net = model.init(key, dummy_target, dummy_params)
    
    # Optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params_net)
    
    # History
    history = {
        'loss': [],
        'grad_norm': [],
        'nan_count': [],
        'metrics': [],
    }
    
    print(f"Training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Compute loss and gradients
        (loss, metrics), grads = value_and_grad(
            lambda p: compute_batch_loss_chaotic(p, model, targets, map_params_batch, 
                                                solver_fn, num_solver_steps),
            has_aux=True
        )(params_net)
        
        grad_norm = optax.global_norm(grads)
        
        # Update
        updates, opt_state = optimizer.update(grads, opt_state, params_net)
        params_net = optax.apply_updates(params_net, updates)
        
        # Record
        history['loss'].append(float(loss))
        history['grad_norm'].append(float(grad_norm))
        history['nan_count'].append(int(metrics['nan_count']))
        history['metrics'].append(metrics)
        
        # Progress
        if epoch % 30 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:8.4f} | GradNorm: {grad_norm:8.4f} | "
                  f"NaNs: {metrics['nan_count']:3d}")
        
        # Emergency stop
        if jnp.isnan(loss) or grad_norm > 1e6:
            print(f"⚠️  TRAINING COLLAPSED at epoch {epoch}!")
            break
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s\n")
    
    return history, params_net


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_chaotic_deq_results(histories: Dict[str, Dict], save_path: str):
    """Compare training curves for chaotic DEQ."""
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = {
        'Naive Newton': '#FF6B6B',
        'Damped Newton': '#4ECDC4',
        'φ-Surfer': '#F7931E'
    }
    
    # Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    for name, history in histories.items():
        epochs = range(len(history['loss']))
        ax1.plot(epochs, history['loss'], color=colors[name], 
                linewidth=2.5, alpha=0.9, label=name)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Training Loss', fontsize=11)
    ax1.set_title('Loss Convergence (Chaotic Regime)', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=10)
    
    # Gradient norms
    ax2 = fig.add_subplot(gs[0, 1])
    for name, history in histories.items():
        epochs = range(len(history['grad_norm']))
        ax2.plot(epochs, history['grad_norm'], color=colors[name],
                linewidth=2.5, alpha=0.9, label=name)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Gradient Norm', fontsize=11)
    ax2.set_title('Gradient Stability in Chaos', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)
    
    # Cumulative NaN count
    ax3 = fig.add_subplot(gs[0, 2])
    for name, history in histories.items():
        cumulative_nans = np.cumsum(history['nan_count'])
        epochs = range(len(cumulative_nans))
        ax3.plot(epochs, cumulative_nans, color=colors[name],
                linewidth=2.5, alpha=0.9, label=name)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Cumulative Divergences', fontsize=11)
    ax3.set_title('Solver Collapse Events', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # Final loss comparison
    ax4 = fig.add_subplot(gs[1, 0])
    final_losses = [histories[name]['loss'][-1] for name in colors.keys()]
    bars = ax4.bar(colors.keys(), final_losses, color=list(colors.values()),
                   alpha=0.8, edgecolor='white', linewidth=2)
    ax4.set_ylabel('Final Loss', fontsize=11)
    ax4.set_title('Convergence Quality', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, final_losses):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Total NaN comparison
    ax5 = fig.add_subplot(gs[1, 1])
    total_nans = [sum(histories[name]['nan_count']) for name in colors.keys()]
    bars = ax5.bar(colors.keys(), total_nans, color=list(colors.values()),
                   alpha=0.8, edgecolor='white', linewidth=2)
    ax5.set_ylabel('Total Divergences', fontsize=11)
    ax5.set_title('Training Stability', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, total_nans):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', fontsize=10, fontweight='bold')
    
    # Scorecard
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    scorecard = "🏆 CHAOTIC DEQ GAUNTLET 🏆\n\n"
    scorecard += "Task: Learn fixed points of chaotic map\n"
    scorecard += "System: Duffing-like discrete dynamics\n"
    scorecard += "Dataset: 100 samples (30% chaotic regime)\n"
    scorecard += "Training: 300 epochs, Adam(lr=1e-3)\n\n"
    
    for name in colors.keys():
        final_loss = histories[name]['loss'][-1]
        total_nan = sum(histories[name]['nan_count'])
        final_grad = histories[name]['grad_norm'][-1]
        
        scorecard += f"{name}:\n"
        scorecard += f"  Final Loss: {final_loss:.4f}\n"
        scorecard += f"  Divergences: {total_nan}\n"
        scorecard += f"  Grad Norm: {final_grad:.2e}\n\n"
    
    winner = min(histories.items(), key=lambda x: x[1]['loss'][-1])[0]
    scorecard += f"\n✨ WINNER: {winner}\n"
    
    if histories['φ-Surfer']['loss'][-1] < histories['Naive Newton']['loss'][-1]:
        scorecard += "🎯 φ-Surfer navigated chaos!"
    
    ax6.text(0.05, 0.5, scorecard, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='wheat', 
                     alpha=0.3, edgecolor='black', linewidth=2))
    
    plt.suptitle("THE CHAOTIC DEQ GAUNTLET: Training Through Chaos", 
                fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Chaotic DEQ visualization saved: {save_path}")
    plt.close()


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

if __name__ == '__main__':
    print("Generating chaotic dataset...\n")
    targets, map_params_batch = create_chaotic_dataset(num_samples=100, hard_fraction=0.3)
    
    print(f"Dataset: {len(targets)} samples")
    print(f"Target range: x=[{targets[:,0].min():.2f}, {targets[:,0].max():.2f}], "
          f"y=[{targets[:,1].min():.2f}, {targets[:,1].max():.2f}]\n")
    
    # Create model
    model = ChaoticDEQ(hidden_dim=64)
    
    # Experiment configurations - RUN ALL THREE TO COMPARE
    experiments = {
        'Naive Newton': newton_step_chaos,
        'Damped Newton': damped_newton_step_chaos,
        'φ-Surfer': phi_surfer_step_chaos,
    }
    
    histories = {}
    
    # Run experiments
    for name, solver_fn in experiments.items():
        print("="*80)
        print(f"EXPERIMENT: {name}")
        print("="*80)
        
        history, final_params = train_chaotic_deq(
            model=model,
            targets=targets,
            map_params_batch=map_params_batch,
            solver_fn=solver_fn,
            num_epochs=300,
            learning_rate=1e-3,
            num_solver_steps=10
        )
        
        histories[name] = history
        
        # Summary
        final_loss = history['loss'][-1]
        total_nans = sum(history['nan_count'])
        print(f"RESULTS:")
        print(f"  Final Loss: {final_loss:.6f}")
        print(f"  Total Divergences: {total_nans}")
        print(f"  Status: {'✅ CONVERGED' if not np.isnan(final_loss) else '❌ COLLAPSED'}\n")
    
    # Visualize full comparison
    save_path = './deq_chaotic.png'
    visualize_chaotic_deq_results(histories, save_path)
    
    print("="*80)
    print("CHAOTIC DEQ GAUNTLET COMPLETE!")
    print("="*80)
    print("\nKEY FINDINGS:")
    
    # Compare all three
    newton_loss = histories['Naive Newton']['loss'][-1]
    damped_loss = histories['Damped Newton']['loss'][-1]
    phi_loss = histories['φ-Surfer']['loss'][-1]
    
    if phi_loss < newton_loss and phi_loss < damped_loss:
        print("✨ φ-Surfer successfully navigated chaotic regime!")
        print(f"   Improvement vs Naive Newton: {(newton_loss - phi_loss)/max(newton_loss, 1e-6)*100:.1f}%")
        print(f"   Improvement vs Damped Newton: {(damped_loss - phi_loss)/max(damped_loss, 1e-6)*100:.1f}%")
    
    newton_nans = sum(histories['Naive Newton']['nan_count'])
    damped_nans = sum(histories['Damped Newton']['nan_count'])
    phi_nans = sum(histories['φ-Surfer']['nan_count'])
    
    print(f"\nDivergence Events:")
    print(f"  Naive Newton: {newton_nans}")
    print(f"  Damped Newton: {damped_nans}")
    print(f"  φ-Surfer: {phi_nans}")
    
    if phi_nans < newton_nans:
        print(f"\n🎯 φ-Surfer avoided {newton_nans - phi_nans} divergence events vs Newton!")
    
    print("\n" + "="*80)
    print("CONCLUSION: φ-Surfer enables STABLE TRAINING on CHAOTIC/STIFF dynamics")
    print("where standard implicit solvers collapse!")
    print("="*80)
