"""
HARD Benchmark: Problems Where Classical Methods Fail

These are scenarios designed to break traditional continuation:
1. Extremely ill-conditioned Jacobians
2. Multiple saddle points
3. Chaotic regions
4. Newton divergence zones

We test if φ-gradient navigation provides robustness.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
from typing import Dict, Tuple, List, Callable
from dataclasses import dataclass

jax.config.update("jax_enable_x64", True)


# =============================================================================
# HARDER TEST FUNCTIONS
# =============================================================================

def wilkinson_like(z: complex, lam: complex) -> complex:
    """
    Wilkinson-like polynomial: Very ill-conditioned near roots.
    f(z) = (z-1)(z-1.001)(z-1.002)...(z-1.01) - λ perturbation
    
    This is pathological because roots are extremely close together.
    """
    product = 1.0 + 0.0j
    for k in range(11):
        product *= (z - (1.0 + 0.001 * k))
    return product - lam * 1e-5


def wilkinson_df_dz(z: complex, lam: complex) -> complex:
    """Derivative via JAX"""
    return jax.grad(lambda z_: jnp.real(wilkinson_like(z_, lam).conj() * wilkinson_like(z_, lam)))(z)


def transcendental(z: complex, lam: complex) -> complex:
    """
    Transcendental equation with multiple branches.
    f(z) = z - λ * sin(z)
    
    Has infinitely many solutions, very sensitive to initial guess.
    """
    return z - lam * jnp.sin(z)


def transcendental_df_dz(z: complex, lam: complex) -> complex:
    """Derivative"""
    return 1.0 - lam * jnp.cos(z)


def double_well(z: complex, lam: complex) -> complex:
    """
    Double-well potential: f(z) = z(z²-1) - λ
    Has saddle point at z=0, two wells at ±1.
    Classical methods get stuck at saddle.
    """
    return z * (z**2 - 1.0) - lam


def double_well_df_dz(z: complex, lam: complex) -> complex:
    """Derivative"""
    return 3.0 * z**2 - 1.0


# =============================================================================
# GENERIC NEWTON ITERATION FOR ANY f
# =============================================================================

def generic_newton_step(z: complex, lam: complex, f_func, df_func) -> complex:
    """Newton step for generic function"""
    fz = f_func(z, lam)
    dfz = df_func(z, lam)
    
    # Safeguard against division by zero
    if jnp.abs(dfz) < 1e-15:
        return z  # Don't move
    
    step = fz / dfz
    
    # Limit step size for stability
    step_size = jnp.abs(step)
    max_step = 0.5
    step = jnp.where(
        step_size > max_step,
        step * (max_step / step_size),
        step
    )
    
    return z - step


# =============================================================================
# φ COMPUTATION FOR GENERIC f
# =============================================================================

def compute_phi_generic(z: complex, 
                       lam: complex,
                       f_func: Callable,
                       df_func: Callable,
                       K: int = 10) -> float:
    """
    Compute φ(z,λ) = -log(min step size in K iterations)
    
    This is the CORE of our method: φ tells us how stable Newton is.
    
    Note: NOT jit-compiled because f_func/df_func vary per problem
    """
    min_step = 1e10
    z_current = z
    
    for k in range(K):
        z_next = generic_newton_step(z_current, lam, f_func, df_func)
        step_size = jnp.abs(z_next - z_current)
        min_step = jnp.minimum(min_step, step_size + 1e-10)
        z_current = z_next
    
    phi = -jnp.log(min_step) * 0.1
    return phi


# =============================================================================
# φ GRADIENT VIA JAX AUTODIFF
# =============================================================================

def make_phi_grad_func(f_func, df_func):
    """
    Factory to create φ gradient function for any f.
    
    This is the MAGIC: JAX differentiates through the entire
    Newton iteration loop!
    
    Note: Cannot JIT this because f_func varies per problem
    """
    def phi_for_grad(z: complex, lam: complex, K: int) -> float:
        return compute_phi_generic(z, lam, f_func, df_func, K)
    
    def phi_grad_z(z: complex, lam: complex, K: int = 10) -> complex:
        # JAX autodiff for complex functions
        # We differentiate φ treating z as two real variables
        def phi_real(z_real, z_imag):
            z_complex = z_real + 1j * z_imag
            return phi_for_grad(z_complex, lam, K)
        
        z_real, z_imag = jnp.real(z), jnp.imag(z)
        
        grad_real = jax.grad(phi_real, argnums=0)(z_real, z_imag)
        grad_imag = jax.grad(phi_real, argnums=1)(z_real, z_imag)
        
        # Wirtinger derivative: ∂φ/∂z = ∂φ/∂x - i∂φ/∂y
        return grad_real - 1j * grad_imag
    
    return phi_grad_z  # Don't JIT - f_func is problem-specific


# =============================================================================
# TEST SCENARIOS
# =============================================================================

@dataclass
class HardProblem:
    name: str
    f: Callable
    df: Callable
    lambda_path: Callable[[float], complex]
    z_init: complex
    description: str
    T: int = 100


# Problem 1: Wilkinson traversal
def wilkinson_path(t: float) -> complex:
    return (1.0 + 5.0 * t) + 0.5j * np.sin(4 * np.pi * t)


HARD_PROBLEMS = [
    HardProblem(
        name="Wilkinson Clusters",
        f=wilkinson_like,
        df=wilkinson_df_dz,
        lambda_path=wilkinson_path,
        z_init=1.005 + 0.0j,  # Start near root cluster
        description="11 roots within 0.01 of each other - extreme ill-conditioning",
        T=80
    ),
    HardProblem(
        name="Double-Well Saddle",
        f=double_well,
        df=double_well_df_dz,
        lambda_path=lambda t: 0.1 * np.exp(2j * np.pi * t),  # Circle around origin
        z_init=0.1 + 0.1j,  # Start near saddle at z=0
        description="Saddle point at origin - Newton gets stuck",
        T=100
    ),
    HardProblem(
        name="Transcendental Maze",
        f=transcendental,
        df=transcendental_df_dz,
        lambda_path=lambda t: 0.5 + 0.5 * t + 0.2j * np.sin(6 * np.pi * t),
        z_init=1.0 + 0.0j,
        description="z - λsin(z) = 0, infinitely many branches",
        T=120
    )
]


# =============================================================================
# BASELINE: Naive Newton Continuation
# =============================================================================

def naive_continuation_generic(problem: HardProblem, noise_std: float = 0.01) -> Dict:
    """Just apply Newton at each step, no prediction"""
    T = problem.T
    time_steps = np.linspace(0, 1, T)
    
    lambda_traj = np.zeros(T, dtype=complex)
    z_traj = np.zeros(T, dtype=complex)
    converged = np.zeros(T, dtype=bool)
    
    z = problem.z_init
    
    for i, t in enumerate(time_steps):
        lam = problem.lambda_path(t)
        
        # Add noise
        z = z + (np.random.randn() + 1j * np.random.randn()) * noise_std
        
        # Newton iteration (5 steps)
        for _ in range(5):
            z = generic_newton_step(z, lam, problem.f, problem.df)
        
        # Check convergence
        residual = np.abs(problem.f(z, lam))
        converged[i] = residual < 1e-6
        
        lambda_traj[i] = lam
        z_traj[i] = z
    
    return {
        'name': 'Naive Newton',
        'lambda': lambda_traj,
        'z': z_traj,
        'converged': converged,
        'success_rate': np.mean(converged)
    }


# =============================================================================
# OUR METHOD: φ-Surfer for Generic Functions
# =============================================================================

def phi_surfer_generic(problem: HardProblem,
                      gamma: float = 0.2,
                      K: int = 15,
                      noise_std: float = 0.01,
                      num_phi_steps: int = 5) -> Dict:
    """
    φ-gradient navigation for ANY differentiable function.
    
    The key advantage: We use φ to sense where Newton will fail,
    and navigate around those regions.
    """
    T = problem.T
    time_steps = np.linspace(0, 1, T)
    
    # Create φ gradient function for this problem
    phi_grad_func = make_phi_grad_func(problem.f, problem.df)
    
    lambda_traj = np.zeros(T, dtype=complex)
    z_traj = np.zeros(T, dtype=complex)
    phi_traj = np.zeros(T)
    converged = np.zeros(T, dtype=bool)
    
    z = problem.z_init
    
    for i, t in enumerate(time_steps):
        lam = problem.lambda_path(t)
        
        # Add noise
        z = z + (np.random.randn() + 1j * np.random.randn()) * noise_std
        
        # Multiple φ-gradient ascent steps
        # This keeps us in regions where Newton is stable
        for _ in range(num_phi_steps):
            g = phi_grad_func(z, lam, K)
            g_norm = jnp.abs(g)
            if g_norm > 1e-12:
                direction = g / g_norm
                z = z + gamma * direction
        
        # Now apply Newton for precision
        for _ in range(3):
            z_new = generic_newton_step(z, lam, problem.f, problem.df)
            # Only accept if converging
            if np.abs(z_new - z) < 0.5:
                z = z_new
            else:
                break
        
        phi_val = compute_phi_generic(z, lam, problem.f, problem.df, K)
        residual = np.abs(problem.f(z, lam))
        converged[i] = residual < 1e-6
        
        lambda_traj[i] = lam
        z_traj[i] = z
        phi_traj[i] = phi_val
    
    return {
        'name': 'φ-Surfer (Ours)',
        'lambda': lambda_traj,
        'z': z_traj,
        'phi': phi_traj,
        'converged': converged,
        'success_rate': np.mean(converged)
    }


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_hard_benchmark(problem: HardProblem, num_trials: int = 5):
    """Compare on hard problems"""
    print(f"\n{'='*80}")
    print(f"HARD PROBLEM: {problem.name}")
    print(f"{'='*80}")
    print(f"Description: {problem.description}")
    print(f"Timesteps: {problem.T}")
    print(f"Trials: {num_trials}")
    print()
    
    methods = [
        ('Naive Newton', naive_continuation_generic),
        ('φ-Surfer (Ours)', phi_surfer_generic)
    ]
    
    results = {}
    
    for method_name, method_func in methods:
        print(f"Running {method_name}...")
        
        trial_results = []
        success_rates = []
        times = []
        
        for trial in range(num_trials):
            np.random.seed(42 + trial)
            
            start = time.time()
            result = method_func(problem, noise_std=0.01)
            elapsed = time.time() - start
            
            trial_results.append(result)
            success_rates.append(result['success_rate'])
            times.append(elapsed)
        
        avg_success = np.mean(success_rates)
        std_success = np.std(success_rates)
        avg_time = np.mean(times)
        
        best_idx = np.argmax(success_rates)
        best_result = trial_results[best_idx]
        
        results[method_name] = {
            'trajectory': best_result,
            'avg_success_rate': avg_success,
            'std_success_rate': std_success,
            'avg_time': avg_time
        }
        
        print(f"  ✓ Success Rate: {avg_success*100:.1f}% ± {std_success*100:.1f}%")
        print(f"    Time: {avg_time:.3f}s")
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_hard_benchmark(problem: HardProblem, results: Dict):
    """Visualize comparison on hard problems"""
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    colors = {
        'Naive Newton': '#FF6B6B',
        'φ-Surfer (Ours)': '#F7931E'
    }
    
    # Plot 1: Parameter Path
    ax1 = fig.add_subplot(gs[0, 0])
    lam_path = results['φ-Surfer (Ours)']['trajectory']['lambda']
    ax1.plot(lam_path.real, lam_path.imag, 'k--', linewidth=2, alpha=0.5)
    ax1.scatter(lam_path[0].real, lam_path[0].imag, c='green', s=100, marker='o')
    ax1.scatter(lam_path[-1].real, lam_path[-1].imag, c='red', s=100, marker='s')
    ax1.set_xlabel('Re(λ)')
    ax1.set_ylabel('Im(λ)')
    ax1.set_title('Parameter Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Trajectories
    ax2 = fig.add_subplot(gs[0, 1])
    for method_name, res in results.items():
        traj = res['trajectory']['z']
        ax2.plot(traj.real, traj.imag, '-', linewidth=2,
                color=colors[method_name], alpha=0.7, label=method_name)
        ax2.scatter(traj[0].real, traj[0].imag, c=colors[method_name], 
                   s=80, marker='o', edgecolors='white', linewidths=2)
    ax2.set_xlabel('Re(z)')
    ax2.set_ylabel('Im(z)')
    ax2.set_title('Solution Trajectories')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Success Rate
    ax3 = fig.add_subplot(gs[0, 2])
    methods = list(results.keys())
    success_means = [results[m]['avg_success_rate'] * 100 for m in methods]
    success_stds = [results[m]['std_success_rate'] * 100 for m in methods]
    bars = ax3.bar(range(len(methods)), success_means,
                   color=[colors[m] for m in methods],
                   yerr=success_stds, capsize=5, alpha=0.8)
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, rotation=15, ha='right')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Convergence Success')
    ax3.set_ylim([0, 105])
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, success_means):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Convergence over time
    ax4 = fig.add_subplot(gs[1, 0])
    for method_name, res in results.items():
        converged = res['trajectory']['converged']
        cumulative = np.cumsum(converged) / np.arange(1, len(converged)+1) * 100
        ax4.plot(cumulative, linewidth=2, color=colors[method_name], label=method_name)
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Cumulative Success Rate (%)')
    ax4.set_title('Convergence Progress')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim([0, 105])
    
    # Plot 5: Residuals
    ax5 = fig.add_subplot(gs[1, 1])
    for method_name, res in results.items():
        z_traj = res['trajectory']['z']
        lam_traj = res['trajectory']['lambda']
        residuals = [np.abs(problem.f(z_traj[i], lam_traj[i])) for i in range(len(z_traj))]
        ax5.semilogy(residuals, linewidth=2, color=colors[method_name], label=method_name)
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('|f(z, λ)|')
    ax5.set_title('Residual (Convergence Quality)')
    ax5.grid(True, alpha=0.3, which='both')
    ax5.legend()
    ax5.axhline(y=1e-6, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Plot 6: φ Landscape (Ours only)
    ax6 = fig.add_subplot(gs[1, 2])
    if 'phi' in results['φ-Surfer (Ours)']['trajectory']:
        phi = results['φ-Surfer (Ours)']['trajectory']['phi']
        ax6.plot(phi, linewidth=2, color=colors['φ-Surfer (Ours)'])
        ax6.set_xlabel('Timestep')
        ax6.set_ylabel('φ Potential')
        ax6.set_title('Stability Landscape (φ)')
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'{problem.name} - {problem.description}',
                fontsize=13, fontweight='bold', y=0.98)
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("HARD BENCHMARK: Problems Where Classical Methods Struggle")
    print("="*80)
    print("\nTesting φ-Surfer on pathological cases:")
    for p in HARD_PROBLEMS:
        print(f"  - {p.name}: {p.description}")
    print()
    
    all_results = {}
    
    for problem in HARD_PROBLEMS:
        results = run_hard_benchmark(problem, num_trials=5)
        all_results[problem.name] = results
        
        fig = visualize_hard_benchmark(problem, results)
        filename = f'/home/midori/gits/3d-newton-fractal/hard_{problem.name.replace(" ", "_")}.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved: {filename}")
        plt.close(fig)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Hard Problems Benchmark")
    print("="*80)
    print(f"{'Problem':<30} {'Naive Newton':<15} {'φ-Surfer (Ours)':<15}")
    print("-"*80)
    
    for prob_name, results in all_results.items():
        naive_sr = results['Naive Newton']['avg_success_rate'] * 100
        phi_sr = results['φ-Surfer (Ours)']['avg_success_rate'] * 100
        print(f"{prob_name:<30} {naive_sr:>6.1f}%          {phi_sr:>6.1f}%")
    
    print("="*80)
    print("\n✅ Hard Benchmark Complete!")
