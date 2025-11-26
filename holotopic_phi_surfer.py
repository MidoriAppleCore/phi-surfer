"""
Holotopic φ-Surfer: Differentiable Root Tracking via Stability Navigation

This implements a novel root-finding agent that navigates parameterized polynomial
systems by "surfing" the stability tube defined by the φ potential field.

Key Innovation: Unlike blind homotopy continuation, this method uses gradient ascent
on φ(z,λ) to actively steer away from fractal boundaries while tracking roots.

Mathematical Foundation:
- System: f(z,λ) = z³ - λ
- Newton map: N_λ(z) = z - f(z,λ)/f'(z,λ)
- Safety potential: φ(z,λ) = -log(min_step + ε)
- Navigation: z_new = z + γ ∇_z φ(z,λ)

Tech Stack: JAX for automatic differentiation through complex Newton iterations
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from typing import Tuple, Callable
import time

# Enable complex number support
jax.config.update("jax_enable_x64", True)


# =============================================================================
# CORE DYNAMICS: Newton Iteration on f(z,λ) = z³ - λ
# =============================================================================

def f(z: complex, lam: complex) -> complex:
    """
    Parameterized cubic: f(z,λ) = z³ - λ
    
    Roots: z* = λ^(1/3) (three cube roots)
    """
    return z**3 - lam


def df_dz(z: complex, lam: complex) -> complex:
    """
    Derivative: ∂f/∂z = 3z²
    """
    return 3 * z**2


def newton_step(z: complex, lam: complex, epsilon: float = 1e-12) -> complex:
    """
    Single Newton iteration: z_new = z - f(z)/f'(z)
    
    Args:
        z: Current position
        lam: Parameter value
        epsilon: Stability term to prevent division by zero
    
    Returns:
        Updated position
    """
    fz = f(z, lam)
    dfz = df_dz(z, lam)
    
    # Stable division with epsilon guard
    step = jnp.where(
        jnp.abs(dfz) > epsilon,
        fz / dfz,
        0.0 + 0.0j  # Don't move if derivative is too small
    )
    
    return z - step


# =============================================================================
# PHI POTENTIAL: Stability Landscape
# =============================================================================

def compute_phi(z: complex, lam: complex, K: int = 10, epsilon: float = 1e-9) -> float:
    """
    Compute the φ safety potential via K-step lookahead.
    
    φ(z,λ) = -log(min_step + ε)
    
    High φ → Deep in convergence basin (safe)
    Low φ  → Near fractal boundary (dangerous)
    
    Args:
        z: Current position
        lam: Parameter value
        K: Lookahead horizon (number of Newton steps)
        epsilon: Numerical stability
    
    Returns:
        φ value (scalar float)
    """
    # Run K Newton steps and track minimum step size
    z_current = z
    min_step = jnp.array(1e5, dtype=jnp.float64)  # Initialize large
    
    for _ in range(K):
        z_next = newton_step(z_current, lam)
        step_size = jnp.abs(z_next - z_current)
        min_step = jnp.minimum(min_step, step_size)
        z_current = z_next
    
    # φ = -log(min_step + ε)
    phi = -jnp.log(min_step + epsilon)
    
    return phi


# JIT compile for speed
compute_phi_jit = jit(compute_phi, static_argnums=(2,))


# =============================================================================
# GRADIENT COMPUTATION: The Magic
# =============================================================================

def phi_grad_z(z: complex, lam: complex, K: int = 10) -> complex:
    """
    Compute ∇_z φ(z,λ) via automatic differentiation.
    
    This is the KEY: JAX backpropagates through the K-step Newton loop
    to give us the exact gradient of the stability landscape.
    
    Args:
        z: Current position (complex)
        lam: Parameter (complex)
        K: Lookahead horizon
    
    Returns:
        Gradient ∇_z φ (complex)
    """
    # JAX requires real inputs for grad, so we split complex → (real, imag)
    def phi_real_imag(z_real: float, z_imag: float) -> float:
        z_complex = z_real + 1j * z_imag
        return compute_phi(z_complex, lam, K)
    
    # Compute gradient w.r.t. real and imaginary parts
    grad_fn = grad(phi_real_imag, argnums=(0, 1))
    z_real, z_imag = z.real, z.imag
    
    grad_real, grad_imag = grad_fn(z_real, z_imag)
    
    # Combine into complex gradient (holomorphic)
    return grad_real + 1j * grad_imag


# JIT compile
phi_grad_z_jit = jit(phi_grad_z, static_argnums=(2,))


# =============================================================================
# HOLOTOPIC SURFER: The Agent
# =============================================================================

def holotopic_update(z: complex, 
                    lam: complex, 
                    gamma: float = 0.05,
                    K: int = 10,
                    do_newton_refine: bool = True) -> Tuple[complex, float]:
    """
    Single update step of the Holotopic φ-Surfer.
    
    Algorithm:
    1. Compute gradient g = ∇_z φ(z,λ)
    2. Move uphill: z' = z + γ * g/|g|
    3. (Optional) Refine with 1 Newton step
    
    Args:
        z: Current position
        lam: Current parameter
        gamma: Step size for gradient ascent
        K: Lookahead horizon for φ
        do_newton_refine: Whether to apply Newton polish
    
    Returns:
        (z_new, phi_value)
    """
    # 1. Sense the landscape
    g = phi_grad_z_jit(z, lam, K)
    phi_val = compute_phi_jit(z, lam, K)
    
    # 2. Normalize gradient and step uphill (toward stability)
    g_norm = jnp.abs(g)
    direction = jnp.where(
        g_norm > 1e-12,
        g / g_norm,
        0.0 + 0.0j
    )
    
    z_new = z + gamma * direction
    
    # 3. Optional Newton refinement (keeps us near actual root)
    if do_newton_refine:
        z_new = newton_step(z_new, lam)
    
    return z_new, phi_val


# JIT compile the full update
holotopic_update_jit = jit(holotopic_update, static_argnums=(3, 4))


# =============================================================================
# PARAMETER PATH: Lambda Evolution
# =============================================================================

def lambda_path(t: float, mode: str = 'spiral') -> complex:
    """
    Define how the parameter λ evolves over time t ∈ [0, 1].
    
    Modes:
    - 'spiral': Exponential spiral outward
    - 'circle': Unit circle rotation
    - 'line': Linear path in complex plane
    
    Args:
        t: Time parameter in [0, 1]
        mode: Path type
    
    Returns:
        λ(t) as complex number
    """
    if mode == 'spiral':
        # Start at 1.0, spiral out while rotating
        radius = 1.0 + 2.0 * t
        angle = 4 * np.pi * t  # Two full rotations
        return radius * np.exp(1j * angle)
    
    elif mode == 'circle':
        # Stay on unit circle, rotate
        angle = 2 * np.pi * t
        return np.exp(1j * angle)
    
    elif mode == 'line':
        # Linear interpolation from 1 to 1+2i
        return 1.0 + 2.0j * t
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# SIMULATION: Full Tracking Experiment
# =============================================================================

def run_tracking_experiment(
    T: int = 200,
    path_mode: str = 'spiral',
    gamma: float = 0.05,
    K: int = 10,
    noise_std: float = 0.02,
    seed: int = 42
) -> dict:
    """
    Run the full Holotopic φ-Surfer tracking experiment.
    
    Setup:
    - Start at λ=1.0, z=1.0 (perfect initialization)
    - Evolve λ along specified path
    - Add random noise to simulate perturbations
    - Track using φ-gradient ascent
    
    Args:
        T: Number of time steps
        path_mode: How λ evolves ('spiral', 'circle', 'line')
        gamma: Gradient ascent step size
        K: Lookahead horizon for φ
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed
    
    Returns:
        Dictionary with trajectory data
    """
    np.random.seed(seed)
    
    # Storage
    time_steps = np.linspace(0, 1, T)
    lambda_traj = np.zeros(T, dtype=complex)
    z_traj = np.zeros(T, dtype=complex)
    phi_traj = np.zeros(T)
    grad_norms = np.zeros(T)
    
    # Initial conditions
    lam = lambda_path(0.0, path_mode)
    z = 1.0 + 0.0j  # Start at a root of λ=1.0
    
    print("🏄 Holotopic φ-Surfer: Starting Experiment")
    print(f"   Path: {path_mode}")
    print(f"   Time steps: {T}")
    print(f"   Gamma: {gamma}")
    print(f"   Lookahead K: {K}")
    print(f"   Noise: σ={noise_std}")
    print()
    
    start_time = time.time()
    
    # Main tracking loop
    for i, t in enumerate(time_steps):
        # Update parameter
        lam = lambda_path(t, path_mode)
        
        # Add perturbation (simulate external drift)
        noise = (np.random.randn() + 1j * np.random.randn()) * noise_std
        z = z + noise
        
        # Holotopic update (gradient ascent on φ)
        z_new, phi_val = holotopic_update_jit(z, lam, gamma, K, True)
        
        # Compute gradient norm for diagnostics
        g = phi_grad_z_jit(z, lam, K)
        grad_norm = np.abs(g)
        
        # Store
        lambda_traj[i] = lam
        z_traj[i] = z_new
        phi_traj[i] = phi_val
        grad_norms[i] = grad_norm
        
        # Update for next iteration
        z = z_new
        
        # Progress
        if i % 50 == 0:
            print(f"   Step {i:3d}/{T}: λ={lam:.3f}, z={z:.3f}, φ={phi_val:.3f}")
    
    elapsed = time.time() - start_time
    print(f"\n✅ Tracking complete in {elapsed:.2f}s")
    print(f"   Final λ: {lam:.4f}")
    print(f"   Final z: {z:.4f}")
    print(f"   Final φ: {phi_val:.4f}")
    
    # Compute tracking error (distance to nearest root)
    z_true = np.power(lam, 1/3)  # Principal cube root (complex-safe)
    error = np.abs(z - z_true)
    print(f"   Tracking error: {error:.6f}")
    
    return {
        'time': time_steps,
        'lambda': lambda_traj,
        'z': z_traj,
        'phi': phi_traj,
        'grad_norms': grad_norms,
        'error': error
    }


# =============================================================================
# VISUALIZATION: Show the Journey
# =============================================================================

def visualize_results(results: dict, path_mode: str):
    """
    Create comprehensive visualization of tracking experiment.
    
    Plots:
    1. Parameter path λ(t) in complex plane
    2. Agent trajectory z(t) overlaid
    3. φ potential landscape (heatmap)
    4. Time series of φ and gradient norm
    """
    fig = plt.figure(figsize=(16, 10))
    
    # --- Plot 1: Trajectories in Complex Plane ---
    ax1 = fig.add_subplot(2, 3, 1)
    
    # Lambda path
    ax1.plot(results['lambda'].real, results['lambda'].imag, 
             'b-', linewidth=2, alpha=0.6, label='λ(t) path')
    ax1.scatter(results['lambda'][0].real, results['lambda'][0].imag,
                c='blue', s=100, marker='o', zorder=5, label='λ start')
    ax1.scatter(results['lambda'][-1].real, results['lambda'][-1].imag,
                c='darkblue', s=100, marker='s', zorder=5, label='λ end')
    
    # Agent trajectory
    ax1.plot(results['z'].real, results['z'].imag,
             'r-', linewidth=1.5, alpha=0.8, label='z(t) agent')
    ax1.scatter(results['z'][0].real, results['z'][0].imag,
                c='red', s=100, marker='o', zorder=5, label='z start')
    ax1.scatter(results['z'][-1].real, results['z'][-1].imag,
                c='darkred', s=100, marker='s', zorder=5, label='z end')
    
    ax1.set_xlabel('Real', fontsize=12)
    ax1.set_ylabel('Imag', fontsize=12)
    ax1.set_title('Parameter Path vs Agent Trajectory', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.axis('equal')
    
    # --- Plot 2: φ Potential Landscape ---
    ax2 = fig.add_subplot(2, 3, 2)
    
    # Create heatmap around final λ
    lam_final = results['lambda'][-1]
    z_final = results['z'][-1]
    
    # Grid for heatmap
    grid_size = 100
    real_range = np.linspace(z_final.real - 1.5, z_final.real + 1.5, grid_size)
    imag_range = np.linspace(z_final.imag - 1.5, z_final.imag + 1.5, grid_size)
    phi_grid = np.zeros((grid_size, grid_size))
    
    print("\n📊 Computing φ landscape (this may take a moment)...")
    for i, re in enumerate(real_range):
        if i % 20 == 0:
            print(f"   Progress: {i}/{grid_size}")
        for j, im in enumerate(imag_range):
            z_test = re + 1j * im
            phi_grid[j, i] = compute_phi_jit(z_test, lam_final, K=10)
    
    # Plot heatmap
    extent = [real_range[0], real_range[-1], imag_range[0], imag_range[-1]]
    im = ax2.imshow(phi_grid, extent=extent, origin='lower', 
                    cmap='viridis', aspect='auto', alpha=0.8)
    
    # Overlay agent path tail
    tail_length = min(50, len(results['z']))
    ax2.plot(results['z'][-tail_length:].real, 
             results['z'][-tail_length:].imag,
             'r-', linewidth=2, alpha=0.6)
    ax2.scatter(z_final.real, z_final.imag, c='red', s=100, 
                marker='*', edgecolors='white', linewidths=2, zorder=10)
    
    plt.colorbar(im, ax=ax2, label='φ(z, λ_final)')
    ax2.set_xlabel('Real', fontsize=12)
    ax2.set_ylabel('Imag', fontsize=12)
    ax2.set_title(f'φ Potential at λ={lam_final:.3f}', fontsize=14, fontweight='bold')
    
    # --- Plot 3: φ Over Time ---
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(results['time'], results['phi'], 'g-', linewidth=2)
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_ylabel('φ value', fontsize=12)
    ax3.set_title('Stability Potential Over Time', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Gradient Norm ---
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(results['time'], results['grad_norms'], 'purple', linewidth=2)
    ax4.set_xlabel('Time', fontsize=12)
    ax4.set_ylabel('||∇φ||', fontsize=12)
    ax4.set_title('Gradient Magnitude', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # --- Plot 5: Tracking Error ---
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Compute error to nearest root at each timestep
    errors = []
    for i in range(len(results['lambda'])):
        lam = results['lambda'][i]
        z = results['z'][i]
        z_true = np.power(lam, 1/3)  # Principal root (complex-safe)
        error = np.abs(z - z_true)
        errors.append(error)
    
    ax5.semilogy(results['time'], errors, 'orange', linewidth=2)
    ax5.set_xlabel('Time', fontsize=12)
    ax5.set_ylabel('|z - z*|', fontsize=12)
    ax5.set_title('Tracking Error (log scale)', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # --- Plot 6: Phase Portrait ---
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Color by time
    scatter = ax6.scatter(results['z'].real, results['z'].imag,
                         c=results['time'], cmap='plasma', s=20, alpha=0.6)
    
    # True roots at final λ
    lam_f = results['lambda'][-1]
    roots = [np.power(lam_f, 1/3) * np.exp(1j * 2 * np.pi * k / 3) for k in range(3)]
    for root in roots:
        ax6.scatter(root.real, root.imag, c='white', s=150, 
                   marker='X', edgecolors='black', linewidths=2, zorder=10)
    
    plt.colorbar(scatter, ax=ax6, label='Time')
    ax6.set_xlabel('Real', fontsize=12)
    ax6.set_ylabel('Imag', fontsize=12)
    ax6.set_title('Agent Phase Portrait', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Holotopic φ-Surfer: {path_mode.capitalize()} Path', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

if __name__ == '__main__':
    print("="*80)
    print("Holotopic φ-Surfer: Differentiable Root Tracking")
    print("="*80)
    print()
    print("This experiment demonstrates navigation of Newton's method using")
    print("gradient ascent on the φ stability potential.")
    print()
    print("Key Features:")
    print("  • JAX automatic differentiation through complex Newton iterations")
    print("  • φ(z,λ) = -log(min_step) encodes basin stability")
    print("  • Agent surfs the stability tube while λ evolves")
    print("  • Robust to noise and perturbations")
    print()
    
    # Run experiment
    results = run_tracking_experiment(
        T=200,              # Time steps
        path_mode='spiral', # Lambda path type
        gamma=0.05,         # Gradient ascent rate
        K=10,               # Lookahead horizon
        noise_std=0.02,     # Perturbation strength
        seed=42
    )
    
    # Visualize
    print("\n📊 Generating visualizations...")
    fig = visualize_results(results, 'spiral')
    
    # Save
    output_path = '/home/midori/gits/3d-newton-fractal/holotopic_surfer_results.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved to: {output_path}")
    
    plt.show()
    
    print("\n" + "="*80)
    print("Experiment Complete!")
    print("="*80)
