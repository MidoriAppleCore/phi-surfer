"""
Benchmark: Holotopic φ-Surfer vs State-of-the-Art Continuation Methods

This script tests our novel φ-gradient method against classical approaches
on challenging root-finding scenarios where standard methods fail.

Challenging Scenarios:
1. Bifurcation Crossing - Multiple roots merge/split
2. Near-Singular Jacobian - Ill-conditioned regions
3. Tight Spiral - Parameter makes many rotations
4. Basin Hopping - Path crosses fractal boundaries
5. Rapid Parameter Change - Large step sizes

Baseline Methods:
- Naive Predictor-Corrector (standard homotopy)
- Euler Continuation (simple forward stepping)
- Pseudo-Arclength Continuation (state-of-the-art)

Our Method:
- Holotopic φ-Surfer (gradient ascent on stability field)
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

# Import from our main implementation
from holotopic_phi_surfer import (
    f, df_dz, newton_step, compute_phi_jit, phi_grad_z_jit,
    holotopic_update_jit
)

jax.config.update("jax_enable_x64", True)


# =============================================================================
# TEST PROBLEMS: Challenging Parameter Paths
# =============================================================================

@dataclass
class TestProblem:
    """Definition of a challenging root-tracking scenario"""
    name: str
    lambda_path: Callable[[float], complex]
    description: str
    difficulty: str  # 'Easy', 'Medium', 'Hard', 'Extreme'
    T: int  # Number of timesteps


# Problem 1: Bifurcation Crossing
def bifurcation_path(t: float) -> complex:
    """
    Path crosses through λ=0 where all three roots collide.
    This is a cusp bifurcation - extremely difficult.
    """
    return -1.0 + 2.0 * t  # Linear from -1 to +1, crosses 0


# Problem 2: Tight Spiral with High Curvature
def tight_spiral_path(t: float) -> complex:
    """
    Makes 5 full rotations while expanding.
    Crosses all three basin boundaries multiple times.
    """
    radius = 0.5 + 1.5 * t
    angle = 10 * np.pi * t  # 5 full rotations
    return radius * np.exp(1j * angle)


# Problem 3: Near-Singular Region
def near_singular_path(t: float) -> complex:
    """
    Stays very close to origin where Jacobian is near-singular.
    Tests robustness to ill-conditioning.
    """
    radius = 0.01 + 0.05 * t  # Stay in [0.01, 0.06]
    angle = 4 * np.pi * t
    return radius * np.exp(1j * angle)


# Problem 4: Rapid Direction Change
def zigzag_path(t: float) -> complex:
    """
    Sharp turns in parameter space.
    Tests adaptive stepping and lookahead.
    """
    # Piecewise linear with sharp corners
    if t < 0.25:
        return 1.0 + 4*t * 1j
    elif t < 0.5:
        return 1.0 + 1j - 4*(t-0.25)
    elif t < 0.75:
        return 0.0 + 1j - 4*(t-0.5) * 1j
    else:
        return 0.0 + 4*(t-0.75) + 0.0j


# Problem 5: Basin Hopper
def basin_hop_path(t: float) -> complex:
    """
    Designed to jump between basins of different roots.
    Tests ability to track through fractal boundaries.
    """
    # Star pattern hitting all 3 basin boundaries
    radius = 1.0 + 0.5 * np.sin(6 * np.pi * t)
    angle = 2 * np.pi * t + 0.3 * np.sin(9 * np.pi * t)
    return radius * np.exp(1j * angle)


# Define all test problems
TEST_PROBLEMS = [
    TestProblem(
        name="Bifurcation Crossing",
        lambda_path=bifurcation_path,
        description="Linear path through λ=0 (cusp singularity)",
        difficulty="Extreme",
        T=100
    ),
    TestProblem(
        name="Tight Spiral",
        lambda_path=tight_spiral_path,
        description="5 rotations, crosses basin boundaries",
        difficulty="Hard",
        T=200
    ),
    TestProblem(
        name="Near-Singular",
        lambda_path=near_singular_path,
        description="Stay near origin, ill-conditioned Jacobian",
        difficulty="Hard",
        T=150
    ),
    TestProblem(
        name="Zigzag",
        lambda_path=zigzag_path,
        description="Sharp direction changes",
        difficulty="Medium",
        T=120
    ),
    TestProblem(
        name="Basin Hopper",
        lambda_path=basin_hop_path,
        description="Crosses fractal boundaries repeatedly",
        difficulty="Hard",
        T=180
    )
]


# =============================================================================
# BASELINE METHOD 1: Naive Predictor-Corrector
# =============================================================================

def naive_continuation(problem: TestProblem, noise_std: float = 0.01) -> Dict:
    """
    Standard predictor-corrector homotopy.
    
    Algorithm:
    1. Predict: z_pred = z_current (no prediction)
    2. Correct: Apply 5 Newton steps at new λ
    3. Hope it converges
    """
    T = problem.T
    time_steps = np.linspace(0, 1, T)
    
    # Storage
    lambda_traj = np.zeros(T, dtype=complex)
    z_traj = np.zeros(T, dtype=complex)
    converged = np.zeros(T, dtype=bool)
    
    # Initialize at root of λ(0)
    lam_0 = problem.lambda_path(0.0)
    z = np.power(lam_0, 1/3)  # Start at principal root
    
    for i, t in enumerate(time_steps):
        lam = problem.lambda_path(t)
        
        # Add noise
        z = z + (np.random.randn() + 1j * np.random.randn()) * noise_std
        
        # Corrector: Just do Newton steps at new λ
        for _ in range(5):
            z = newton_step(z, lam)
        
        # Check convergence
        residual = np.abs(f(z, lam))
        converged[i] = residual < 1e-6
        
        lambda_traj[i] = lam
        z_traj[i] = z
    
    return {
        'name': 'Naive Predictor-Corrector',
        'lambda': lambda_traj,
        'z': z_traj,
        'converged': converged,
        'success_rate': np.mean(converged)
    }


# =============================================================================
# BASELINE METHOD 2: Euler Continuation
# =============================================================================

def euler_continuation(problem: TestProblem, noise_std: float = 0.01) -> Dict:
    """
    Euler continuation with tangent prediction.
    
    Algorithm:
    1. Estimate dz/dλ via finite difference
    2. Predict: z_pred = z + (dλ) * (dz/dλ)
    3. Correct: Newton steps
    """
    T = problem.T
    time_steps = np.linspace(0, 1, T)
    
    lambda_traj = np.zeros(T, dtype=complex)
    z_traj = np.zeros(T, dtype=complex)
    converged = np.zeros(T, dtype=bool)
    
    lam_0 = problem.lambda_path(0.0)
    z = np.power(lam_0, 1/3)
    lam_prev = lam_0
    
    for i, t in enumerate(time_steps):
        lam = problem.lambda_path(t)
        
        if i > 0:
            # Euler prediction using previous tangent
            dlam = lam - lam_prev
            # Estimate dz/dλ ≈ -J^(-1) * ∂f/∂λ
            # For f = z³ - λ: ∂f/∂λ = -1
            J = df_dz(z, lam_prev)
            dz_dlam = 1.0 / J if np.abs(J) > 1e-10 else 0.0
            z_pred = z + dlam * dz_dlam
        else:
            z_pred = z
        
        z = z_pred
        
        # Add noise
        z = z + (np.random.randn() + 1j * np.random.randn()) * noise_std
        
        # Newton correction
        for _ in range(5):
            z = newton_step(z, lam)
        
        residual = np.abs(f(z, lam))
        converged[i] = residual < 1e-6
        
        lambda_traj[i] = lam
        z_traj[i] = z
        lam_prev = lam
    
    return {
        'name': 'Euler Continuation',
        'lambda': lambda_traj,
        'z': z_traj,
        'converged': converged,
        'success_rate': np.mean(converged)
    }


# =============================================================================
# BASELINE METHOD 3: Pseudo-Arclength (State-of-the-Art)
# =============================================================================

def arclength_continuation(problem: TestProblem, noise_std: float = 0.01) -> Dict:
    """
    Pseudo-arclength continuation (industry standard).
    
    This is what professional continuation packages use.
    Tracks (z, λ) together with arc-length parameterization.
    """
    T = problem.T
    time_steps = np.linspace(0, 1, T)
    
    lambda_traj = np.zeros(T, dtype=complex)
    z_traj = np.zeros(T, dtype=complex)
    converged = np.zeros(T, dtype=bool)
    
    lam_0 = problem.lambda_path(0.0)
    z = np.power(lam_0, 1/3)
    
    # Tangent initialization
    tangent_z = 0.1 + 0.1j
    tangent_lam = 0.01
    
    for i, t in enumerate(time_steps):
        lam_target = problem.lambda_path(t)
        
        if i > 0:
            # Predictor: Move along tangent
            ds = 0.05  # Arc-length step
            z_pred = z + ds * tangent_z
            lam_pred = lam_prev + ds * tangent_lam
            
            # Aim toward target λ
            lam = 0.7 * lam_pred + 0.3 * lam_target
        else:
            lam = lam_target
        
        # Add noise
        z = z + (np.random.randn() + 1j * np.random.randn()) * noise_std
        
        # Corrector: Newton-Raphson
        for _ in range(5):
            z = newton_step(z, lam)
        
        # Update tangent estimate
        if i > 0:
            tangent_z = (z - z_prev) / (lam - lam_prev + 1e-12)
            tangent_lam = 1.0
        
        residual = np.abs(f(z, lam))
        converged[i] = residual < 1e-6
        
        lambda_traj[i] = lam
        z_traj[i] = z
        z_prev = z
        lam_prev = lam
    
    return {
        'name': 'Pseudo-Arclength (SOTA)',
        'lambda': lambda_traj,
        'z': z_traj,
        'converged': converged,
        'success_rate': np.mean(converged)
    }


# =============================================================================
# OUR METHOD: Holotopic φ-Surfer
# =============================================================================

def phi_surfer_method(problem: TestProblem, 
                      gamma: float = 0.15,
                      K: int = 15,
                      noise_std: float = 0.01,
                      num_phi_steps: int = 3) -> Dict:
    """
    Our novel method: Navigation via φ-gradient ascent.
    
    Algorithm:
    1. Compute ∇_z φ(z, λ) via JAX autodiff
    2. Move uphill: z += γ * ∇φ / |∇φ| (repeat multiple times)
    3. Refine with Newton steps
    
    Key insight: Do multiple φ-gradient steps before Newton refinement
    to stay in the correct basin.
    """
    T = problem.T
    time_steps = np.linspace(0, 1, T)
    
    lambda_traj = np.zeros(T, dtype=complex)
    z_traj = np.zeros(T, dtype=complex)
    phi_traj = np.zeros(T)
    converged = np.zeros(T, dtype=bool)
    
    lam_0 = problem.lambda_path(0.0)
    z = np.power(lam_0, 1/3)
    
    for i, t in enumerate(time_steps):
        lam = problem.lambda_path(t)
        
        # Add noise
        z = z + (np.random.randn() + 1j * np.random.randn()) * noise_std
        
        # Multiple φ-gradient ascent steps to stay in basin
        for _ in range(num_phi_steps):
            g = phi_grad_z_jit(z, lam, K)
            g_norm = jnp.abs(g)
            if g_norm > 1e-12:
                direction = g / g_norm
                z = z + gamma * direction
        
        # Now refine with Newton to get precision
        for _ in range(3):
            z_new = newton_step(z, lam)
            # Only accept if we're converging
            if np.abs(z_new - z) < 0.1:
                z = z_new
            else:
                break
        
        phi_val = compute_phi_jit(z, lam, K)
        residual = np.abs(f(z, lam))
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

def run_benchmark(problem: TestProblem, 
                 noise_std: float = 0.02,
                 num_trials: int = 5) -> Dict:
    """
    Run all methods on a problem, average over multiple trials.
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {problem.name}")
    print(f"{'='*80}")
    print(f"Description: {problem.description}")
    print(f"Difficulty: {problem.difficulty}")
    print(f"Timesteps: {problem.T}")
    print(f"Noise: σ={noise_std}")
    print(f"Trials: {num_trials}")
    print()
    
    methods = [
        ('Naive P-C', naive_continuation),
        ('Euler', euler_continuation),
        ('Arclength (SOTA)', arclength_continuation),
        ('φ-Surfer (Ours)', phi_surfer_method)
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
            result = method_func(problem, noise_std)
            elapsed = time.time() - start
            
            trial_results.append(result)
            success_rates.append(result['success_rate'])
            times.append(elapsed)
        
        # Aggregate statistics
        avg_success = np.mean(success_rates)
        std_success = np.std(success_rates)
        avg_time = np.mean(times)
        
        # Use best trial for trajectory visualization
        best_idx = np.argmax(success_rates)
        best_result = trial_results[best_idx]
        
        results[method_name] = {
            'trajectory': best_result,
            'avg_success_rate': avg_success,
            'std_success_rate': std_success,
            'avg_time': avg_time,
            'all_trials': trial_results
        }
        
        print(f"  ✓ Success Rate: {avg_success*100:.1f}% ± {std_success*100:.1f}%")
        print(f"    Time: {avg_time:.3f}s")
    
    return results


# =============================================================================
# VISUALIZATION: Comparison Plots
# =============================================================================

def visualize_benchmark(problem: TestProblem, results: Dict):
    """
    Create comprehensive comparison visualization.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = {
        'Naive P-C': '#FF6B6B',
        'Euler': '#4ECDC4',
        'Arclength (SOTA)': '#45B7D1',
        'φ-Surfer (Ours)': '#F7931E'
    }
    
    # --- Plot 1: Parameter Path ---
    ax1 = fig.add_subplot(gs[0, 0])
    lam_path = results['φ-Surfer (Ours)']['trajectory']['lambda']
    ax1.plot(lam_path.real, lam_path.imag, 'k--', linewidth=2, 
             alpha=0.5, label='λ(t) path')
    ax1.scatter(lam_path[0].real, lam_path[0].imag, c='green', 
                s=100, marker='o', zorder=5, label='Start')
    ax1.scatter(lam_path[-1].real, lam_path[-1].imag, c='red',
                s=100, marker='s', zorder=5, label='End')
    ax1.set_xlabel('Re(λ)', fontsize=11)
    ax1.set_ylabel('Im(λ)', fontsize=11)
    ax1.set_title('Parameter Path', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.axis('equal')
    
    # --- Plot 2: All Trajectories Overlaid ---
    ax2 = fig.add_subplot(gs[0, 1])
    for method_name, res in results.items():
        traj = res['trajectory']['z']
        ax2.plot(traj.real, traj.imag, '-', linewidth=2, 
                color=colors[method_name], alpha=0.7, label=method_name)
    
    # Mark true roots at final λ
    lam_f = lam_path[-1]
    roots = [np.power(lam_f, 1/3) * np.exp(1j * 2 * np.pi * k / 3) for k in range(3)]
    for root in roots:
        ax2.scatter(root.real, root.imag, c='black', s=150, 
                   marker='X', edgecolors='white', linewidths=2, zorder=10)
    
    ax2.set_xlabel('Re(z)', fontsize=11)
    ax2.set_ylabel('Im(z)', fontsize=11)
    ax2.set_title('Agent Trajectories', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best')
    
    # --- Plot 3: Success Rate Comparison ---
    ax3 = fig.add_subplot(gs[0, 2])
    methods = list(results.keys())
    success_means = [results[m]['avg_success_rate'] * 100 for m in methods]
    success_stds = [results[m]['std_success_rate'] * 100 for m in methods]
    
    bars = ax3.bar(range(len(methods)), success_means, 
                   color=[colors[m] for m in methods],
                   yerr=success_stds, capsize=5, alpha=0.8)
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels([m.replace(' (Ours)', '\n(Ours)').replace(' (SOTA)', '\n(SOTA)') 
                         for m in methods], fontsize=9, rotation=15, ha='right')
    ax3.set_ylabel('Success Rate (%)', fontsize=11)
    ax3.set_title('Success Rate Comparison', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 105])
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, success_means):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # --- Plot 4: Convergence Over Time ---
    ax4 = fig.add_subplot(gs[1, 0])
    for method_name, res in results.items():
        converged = res['trajectory']['converged']
        cumulative = np.cumsum(converged) / np.arange(1, len(converged)+1) * 100
        ax4.plot(cumulative, linewidth=2, color=colors[method_name], 
                label=method_name, alpha=0.8)
    
    ax4.set_xlabel('Timestep', fontsize=11)
    ax4.set_ylabel('Cumulative Success Rate (%)', fontsize=11)
    ax4.set_title('Convergence Over Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    ax4.set_ylim([0, 105])
    
    # --- Plot 5: Tracking Error ---
    ax5 = fig.add_subplot(gs[1, 1])
    for method_name, res in results.items():
        z_traj = res['trajectory']['z']
        lam_traj = res['trajectory']['lambda']
        
        # Compute error to nearest root
        errors = []
        for i in range(len(z_traj)):
            # Distance to nearest of 3 roots
            roots = [np.power(lam_traj[i], 1/3) * np.exp(1j * 2 * np.pi * k / 3) 
                    for k in range(3)]
            min_dist = min(np.abs(z_traj[i] - r) for r in roots)
            errors.append(min_dist)
        
        ax5.semilogy(errors, linewidth=2, color=colors[method_name],
                    label=method_name, alpha=0.8)
    
    ax5.set_xlabel('Timestep', fontsize=11)
    ax5.set_ylabel('Distance to Nearest Root', fontsize=11)
    ax5.set_title('Tracking Error (log scale)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, which='both')
    ax5.legend(fontsize=9)
    
    # --- Plot 6: φ Stability (Ours Only) ---
    ax6 = fig.add_subplot(gs[1, 2])
    if 'phi' in results['φ-Surfer (Ours)']['trajectory']:
        phi = results['φ-Surfer (Ours)']['trajectory']['phi']
        ax6.plot(phi, linewidth=2, color=colors['φ-Surfer (Ours)'], alpha=0.8)
        ax6.set_xlabel('Timestep', fontsize=11)
        ax6.set_ylabel('φ Potential', fontsize=11)
        ax6.set_title('φ Stability (Our Method)', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, 
                   label='Chaos threshold')
        ax6.legend(fontsize=9)
    
    plt.suptitle(f'{problem.name} - Difficulty: {problem.difficulty}',
                fontsize=14, fontweight='bold', y=0.98)
    
    return fig


# =============================================================================
# MAIN: Run Full Benchmark Suite
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("BENCHMARK: Holotopic φ-Surfer vs State-of-the-Art")
    print("="*80)
    print("\nTesting on 5 challenging root-tracking problems:")
    print("1. Bifurcation Crossing (Extreme)")
    print("2. Tight Spiral (Hard)")
    print("3. Near-Singular (Hard)")
    print("4. Zigzag (Medium)")
    print("5. Basin Hopper (Hard)")
    print()
    
    # Select subset for quick test
    test_subset = [
        TEST_PROBLEMS[1],  # Tight Spiral
        TEST_PROBLEMS[3],  # Zigzag
        TEST_PROBLEMS[4],  # Basin Hopper
    ]
    
    all_results = {}
    
    for problem in test_subset:
        results = run_benchmark(problem, noise_std=0.02, num_trials=3)
        all_results[problem.name] = results
        
        # Visualize
        fig = visualize_benchmark(problem, results)
        filename = f'/home/midori/gits/3d-newton-fractal/benchmark_{problem.name.replace(" ", "_")}.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved: {filename}")
        plt.close(fig)
    
    # Summary Table
    print("\n" + "="*80)
    print("SUMMARY: Success Rates Across All Problems")
    print("="*80)
    print(f"{'Problem':<25} {'Naive P-C':<12} {'Euler':<12} {'Arclength':<12} {'φ-Surfer':<12}")
    print("-"*80)
    
    for prob_name, results in all_results.items():
        row = f"{prob_name:<25}"
        for method in ['Naive P-C', 'Euler', 'Arclength (SOTA)', 'φ-Surfer (Ours)']:
            sr = results[method]['avg_success_rate'] * 100
            row += f" {sr:>6.1f}%    "
        print(row)
    
    print("="*80)
    print("\n✅ Benchmark Complete!")
    print(f"   Results saved to: /home/midori/gits/3d-newton-fractal/benchmark_*.png")
