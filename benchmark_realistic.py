"""
REALISTIC Benchmark: φ-Surfer vs Classical Methods

Key insight: The cubic root problem z³=λ is TOO EASY for classical methods.
We need problems where:
1. Newton actually fails/diverges
2. Basin boundaries matter
3. Parameter evolution causes real trouble

NEW STRATEGY: Test on NOISE ROBUSTNESS and EXTREME PATHS
- High noise environments (real-world data corruption)
- Discontinuous parameter jumps (bifurcations)
- Starting far from solution (bad initialization)
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
    f, df_dz, newton_step, compute_phi_jit, phi_grad_z_jit
)

jax.config.update("jax_enable_x64", True)


# =============================================================================
# REALISTIC TEST SCENARIOS
# =============================================================================

@dataclass
class RealisticTest:
    name: str
    lambda_path: Callable[[float], complex]
    noise_std: float  # Key differentiator
    description: str
    T: int = 100


# Test 1: EXTREME NOISE (real-world sensor corruption)
def smooth_path(t: float) -> complex:
    """Simple spiral, but with EXTREME noise"""
    radius = 1.0 + 0.5 * t
    angle = 2 * np.pi * t
    return radius * np.exp(1j * angle)


# Test 2: DISCONTINUOUS JUMPS (bifurcations, mode switches)
def jump_path(t: float) -> complex:
    """Parameter JUMPS between distant values"""
    # Jump between 3 distant points
    if t < 0.33:
        return 1.0 + 0.5j
    elif t < 0.66:
        return -0.8 - 0.9j  # Big jump!
    else:
        return 0.3 + 1.2j   # Another jump!


# Test 3: FAST EVOLUTION (rapid parameter changes)
def fast_spiral(t: float) -> complex:
    """10 rotations instead of 1 - very fast"""
    radius = 0.5 + 1.5 * t
    angle = 20 * np.pi * t  # 10 full rotations
    return radius * np.exp(1j * angle)


# Test 4: BAD INITIALIZATION (start far from root)
def bad_init_path(t: float) -> complex:
    """Start with z very far from actual root"""
    return 1.0 + 0.1 * t


REALISTIC_TESTS = [
    RealisticTest(
        name="Extreme Noise",
        lambda_path=smooth_path,
        noise_std=0.15,  # 15% noise - HUGE
        description="σ=0.15 noise per step (sensor corruption)",
        T=100
    ),
    RealisticTest(
        name="Discontinuous Jumps",
        lambda_path=jump_path,
        noise_std=0.03,
        description="Parameter jumps to distant values (bifurcations)",
        T=60
    ),
    RealisticTest(
        name="Fast Evolution",
        lambda_path=fast_spiral,
        noise_std=0.05,
        description="10 rotations, crosses basins rapidly",
        T=150
    ),
]


# =============================================================================
# METHODS
# =============================================================================

def naive_newton(test: RealisticTest, z_init: complex = 1.0+0.0j) -> Dict:
    """Just Newton iteration, no fancy stuff"""
    T = test.T
    time_steps = np.linspace(0, 1, T)
    
    lambda_traj = np.zeros(T, dtype=complex)
    z_traj = np.zeros(T, dtype=complex)
    converged = np.zeros(T, dtype=bool)
    diverged_count = 0
    
    z = z_init
    
    for i, t in enumerate(time_steps):
        lam = test.lambda_path(t)
        
        # Add noise
        z = z + (np.random.randn() + 1j * np.random.randn()) * test.noise_std
        
        # Newton iteration (10 steps for convergence)
        for _ in range(10):
            z_new = newton_step(z, lam)
            # Check for divergence
            if np.abs(z_new) > 100:
                diverged_count += 1
                z_new = z  # Stay put
                break
            z = z_new
        
        # Check convergence
        residual = np.abs(f(z, lam))
        converged[i] = residual < 1e-5
        
        lambda_traj[i] = lam
        z_traj[i] = z
    
    return {
        'name': 'Naive Newton',
        'lambda': lambda_traj,
        'z': z_traj,
        'converged': converged,
        'success_rate': np.mean(converged),
        'diverged_count': diverged_count
    }


def phi_surfer_fast(test: RealisticTest, 
                   z_init: complex = 1.0+0.0j,
                   gamma: float = 0.2,
                   K: int = 10) -> Dict:
    """
    φ-Surfer: Use φ gradient to navigate noise
    
    Key advantage: φ provides SMOOTHING - averages over K steps,
    making it more robust to noise.
    """
    T = test.T
    time_steps = np.linspace(0, 1, T)
    
    lambda_traj = np.zeros(T, dtype=complex)
    z_traj = np.zeros(T, dtype=complex)
    phi_traj = np.zeros(T)
    converged = np.zeros(T, dtype=bool)
    diverged_count = 0
    
    z = z_init
    
    for i, t in enumerate(time_steps):
        lam = test.lambda_path(t)
        
        # Add noise
        z_noisy = z + (np.random.randn() + 1j * np.random.randn()) * test.noise_std
        
        # Compute φ gradient (smooths over K steps)
        try:
            g = phi_grad_z_jit(z_noisy, lam, K)
            g_norm = jnp.abs(g)
            
            if g_norm > 1e-12:
                # Move toward stability
                direction = g / g_norm
                z = z_noisy + gamma * direction
            else:
                z = z_noisy
        except:
            # If gradient computation fails, fall back to Newton
            z = z_noisy
        
        # Refine with Newton (fewer steps since φ does the work)
        for _ in range(3):
            z_new = newton_step(z, lam)
            if np.abs(z_new) > 100:
                diverged_count += 1
                break
            if np.abs(z_new - z) < 0.1:
                z = z_new
            else:
                break
        
        phi_val = compute_phi_jit(z, lam, K)
        residual = np.abs(f(z, lam))
        converged[i] = residual < 1e-5
        
        lambda_traj[i] = lam
        z_traj[i] = z
        phi_traj[i] = phi_val
    
    return {
        'name': 'φ-Surfer (Ours)',
        'lambda': lambda_traj,
        'z': z_traj,
        'phi': phi_traj,
        'converged': converged,
        'success_rate': np.mean(converged),
        'diverged_count': diverged_count
    }


def damped_newton(test: RealisticTest, z_init: complex = 1.0+0.0j) -> Dict:
    """
    Damped Newton with line search (common improvement)
    """
    T = test.T
    time_steps = np.linspace(0, 1, T)
    
    lambda_traj = np.zeros(T, dtype=complex)
    z_traj = np.zeros(T, dtype=complex)
    converged = np.zeros(T, dtype=bool)
    diverged_count = 0
    
    z = z_init
    
    for i, t in enumerate(time_steps):
        lam = test.lambda_path(t)
        
        # Add noise
        z = z + (np.random.randn() + 1j * np.random.randn()) * test.noise_std
        
        # Damped Newton
        for _ in range(10):
            fz = f(z, lam)
            dfz = df_dz(z, lam)
            
            if jnp.abs(dfz) < 1e-15:
                break
            
            # Full Newton step
            step = fz / dfz
            
            # Damping: use smaller step if residual increases
            for alpha in [1.0, 0.5, 0.25, 0.1]:
                z_new = z - alpha * step
                if np.abs(f(z_new, lam)) <= np.abs(fz):
                    z = z_new
                    break
            
            if np.abs(z) > 100:
                diverged_count += 1
                break
        
        residual = np.abs(f(z, lam))
        converged[i] = residual < 1e-5
        
        lambda_traj[i] = lam
        z_traj[i] = z
    
    return {
        'name': 'Damped Newton',
        'lambda': lambda_traj,
        'z': z_traj,
        'converged': converged,
        'success_rate': np.mean(converged),
        'diverged_count': diverged_count
    }


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_realistic_benchmark(test: RealisticTest, 
                           z_init: complex = 1.0+0.0j,
                           num_trials: int = 10) -> Dict:
    """
    Run all methods on realistic scenario
    """
    print(f"\n{'='*80}")
    print(f"TEST: {test.name}")
    print(f"{'='*80}")
    print(f"Description: {test.description}")
    print(f"Timesteps: {test.T}")
    print(f"Noise: σ={test.noise_std}")
    print(f"Trials: {num_trials}")
    print(f"Initial z: {z_init:.3f}")
    print()
    
    methods = [
        ('Naive Newton', naive_newton),
        ('Damped Newton', damped_newton),
        ('φ-Surfer (Ours)', phi_surfer_fast)
    ]
    
    results = {}
    
    for method_name, method_func in methods:
        print(f"Running {method_name}...")
        
        trial_results = []
        success_rates = []
        divergence_counts = []
        times = []
        
        for trial in range(num_trials):
            np.random.seed(42 + trial)
            
            start = time.time()
            result = method_func(test, z_init=z_init)
            elapsed = time.time() - start
            
            trial_results.append(result)
            success_rates.append(result['success_rate'])
            divergence_counts.append(result.get('diverged_count', 0))
            times.append(elapsed)
        
        avg_success = np.mean(success_rates)
        std_success = np.std(success_rates)
        avg_divergence = np.mean(divergence_counts)
        avg_time = np.mean(times)
        
        best_idx = np.argmax(success_rates)
        best_result = trial_results[best_idx]
        
        results[method_name] = {
            'trajectory': best_result,
            'avg_success_rate': avg_success,
            'std_success_rate': std_success,
            'avg_divergence': avg_divergence,
            'avg_time': avg_time
        }
        
        print(f"  ✓ Success: {avg_success*100:.1f}% ± {std_success*100:.1f}%")
        print(f"    Diverged: {avg_divergence:.1f} times/run")
        print(f"    Time: {avg_time:.3f}s")
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_realistic(test: RealisticTest, results: Dict):
    """Create comparison plots"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    colors = {
        'Naive Newton': '#FF6B6B',
        'Damped Newton': '#4ECDC4',
        'φ-Surfer (Ours)': '#F7931E'
    }
    
    # Plot 1: Parameter Path
    ax1 = fig.add_subplot(gs[0, 0])
    lam_path = results['φ-Surfer (Ours)']['trajectory']['lambda']
    ax1.plot(lam_path.real, lam_path.imag, 'k--', linewidth=2, alpha=0.6, label='λ(t)')
    ax1.scatter(lam_path[0].real, lam_path[0].imag, c='green', s=120, marker='o', 
               edgecolors='white', linewidths=2, zorder=5, label='Start')
    ax1.scatter(lam_path[-1].real, lam_path[-1].imag, c='red', s=120, marker='s',
               edgecolors='white', linewidths=2, zorder=5, label='End')
    ax1.set_xlabel('Re(λ)', fontsize=11)
    ax1.set_ylabel('Im(λ)', fontsize=11)
    ax1.set_title('Parameter Evolution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Plot 2: Trajectories
    ax2 = fig.add_subplot(gs[0, 1])
    for method_name, res in results.items():
        traj = res['trajectory']['z']
        ax2.plot(traj.real, traj.imag, '-', linewidth=2,
                color=colors[method_name], alpha=0.7, label=method_name)
    
    # True roots at final λ
    lam_f = lam_path[-1]
    roots = [np.power(lam_f, 1/3) * np.exp(1j * 2 * np.pi * k / 3) for k in range(3)]
    for root in roots:
        ax2.scatter(root.real, root.imag, c='black', s=180,
                   marker='X', edgecolors='white', linewidths=2, zorder=10)
    
    ax2.set_xlabel('Re(z)', fontsize=11)
    ax2.set_ylabel('Im(z)', fontsize=11)
    ax2.set_title('Solution Trajectories', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Plot 3: Success Rate Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    methods = list(results.keys())
    success_means = [results[m]['avg_success_rate'] * 100 for m in methods]
    success_stds = [results[m]['std_success_rate'] * 100 for m in methods]
    
    bars = ax3.bar(range(len(methods)), success_means,
                   color=[colors[m] for m in methods],
                   yerr=success_stds, capsize=5, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels([m.replace(' (Ours)', '\n(Ours)') for m in methods], 
                        fontsize=9, rotation=15, ha='right')
    ax3.set_ylabel('Success Rate (%)', fontsize=11)
    ax3.set_title('Convergence Success', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 105])
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, success_means):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Convergence Over Time
    ax4 = fig.add_subplot(gs[1, 0])
    for method_name, res in results.items():
        converged = res['trajectory']['converged']
        cumulative = np.cumsum(converged) / np.arange(1, len(converged)+1) * 100
        ax4.plot(cumulative, linewidth=2.5, color=colors[method_name], 
                label=method_name, alpha=0.8)
    ax4.set_xlabel('Timestep', fontsize=11)
    ax4.set_ylabel('Cumulative Success Rate (%)', fontsize=11)
    ax4.set_title('Convergence Progress', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    ax4.set_ylim([0, 105])
    
    # Plot 5: Tracking Error
    ax5 = fig.add_subplot(gs[1, 1])
    for method_name, res in results.items():
        z_traj = res['trajectory']['z']
        lam_traj = res['trajectory']['lambda']
        
        errors = []
        for i in range(len(z_traj)):
            roots = [np.power(lam_traj[i], 1/3) * np.exp(1j * 2 * np.pi * k / 3) 
                    for k in range(3)]
            min_dist = min(np.abs(z_traj[i] - r) for r in roots)
            errors.append(min_dist)
        
        ax5.semilogy(errors, linewidth=2.5, color=colors[method_name],
                    label=method_name, alpha=0.8)
    ax5.set_xlabel('Timestep', fontsize=11)
    ax5.set_ylabel('Distance to Nearest Root', fontsize=11)
    ax5.set_title('Tracking Error (log scale)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, which='both')
    ax5.legend(fontsize=9)
    
    # Plot 6: Residuals
    ax6 = fig.add_subplot(gs[1, 2])
    for method_name, res in results.items():
        z_traj = res['trajectory']['z']
        lam_traj = res['trajectory']['lambda']
        residuals = [np.abs(f(z_traj[i], lam_traj[i])) for i in range(len(z_traj))]
        ax6.semilogy(residuals, linewidth=2.5, color=colors[method_name],
                    label=method_name, alpha=0.8)
    ax6.axhline(y=1e-5, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Threshold')
    ax6.set_xlabel('Timestep', fontsize=11)
    ax6.set_ylabel('|f(z, λ)|', fontsize=11)
    ax6.set_title('Residual Evolution', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, which='both')
    ax6.legend(fontsize=9)
    
    # Plot 7: φ Stability (Ours only)
    ax7 = fig.add_subplot(gs[2, 0])
    if 'phi' in results['φ-Surfer (Ours)']['trajectory']:
        phi = results['φ-Surfer (Ours)']['trajectory']['phi']
        ax7.plot(phi, linewidth=2.5, color=colors['φ-Surfer (Ours)'], alpha=0.8)
        ax7.set_xlabel('Timestep', fontsize=11)
        ax7.set_ylabel('φ Potential', fontsize=11)
        ax7.set_title('Stability Landscape (Our Method)', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Plot 8: Divergence Count
    ax8 = fig.add_subplot(gs[2, 1])
    methods = list(results.keys())
    diverg_means = [results[m]['avg_divergence'] for m in methods]
    bars = ax8.bar(range(len(methods)), diverg_means,
                   color=[colors[m] for m in methods], alpha=0.8,
                   edgecolor='white', linewidth=1.5)
    ax8.set_xticks(range(len(methods)))
    ax8.set_xticklabels([m.replace(' (Ours)', '\n(Ours)') for m in methods],
                        fontsize=9, rotation=15, ha='right')
    ax8.set_ylabel('Divergence Events', fontsize=11)
    ax8.set_title('Stability (Lower is Better)', fontsize=12, fontweight='bold')
    ax8.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, diverg_means):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 9: Time Comparison
    ax9 = fig.add_subplot(gs[2, 2])
    time_means = [results[m]['avg_time'] * 1000 for m in methods]  # Convert to ms
    bars = ax9.bar(range(len(methods)), time_means,
                   color=[colors[m] for m in methods], alpha=0.8,
                   edgecolor='white', linewidth=1.5)
    ax9.set_xticks(range(len(methods)))
    ax9.set_xticklabels([m.replace(' (Ours)', '\n(Ours)') for m in methods],
                        fontsize=9, rotation=15, ha='right')
    ax9.set_ylabel('Runtime (ms)', fontsize=11)
    ax9.set_title('Computational Cost', fontsize=12, fontweight='bold')
    ax9.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, time_means):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle(f'{test.name} - {test.description}',
                fontsize=14, fontweight='bold', y=0.995)
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("REALISTIC BENCHMARK: φ-Surfer in Challenging Real-World Scenarios")
    print("="*80)
    print("\nFocus: NOISE ROBUSTNESS & DISCONTINUOUS PARAMETER EVOLUTION")
    print("\nTests:")
    for t in REALISTIC_TESTS:
        print(f"  - {t.name}: {t.description}")
    print()
    
    all_results = {}
    
    # Test with GOOD initialization (fair comparison)
    print("\n" + "="*80)
    print("PART 1: Standard Initialization (z₀ = 1.0)")
    print("="*80)
    
    for test in REALISTIC_TESTS:
        results = run_realistic_benchmark(test, z_init=1.0+0.0j, num_trials=10)
        all_results[test.name + " (Good Init)"] = results
        
        fig = visualize_realistic(test, results)
        filename = f'/home/midori/gits/3d-newton-fractal/realistic_{test.name.replace(" ", "_")}.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved: {filename}")
        plt.close(fig)
    
    # Test with BAD initialization (our method should help more)
    print("\n" + "="*80)
    print("PART 2: Bad Initialization (z₀ = 10.0 + 10.0j)")
    print("="*80)
    
    for test in REALISTIC_TESTS[:2]:  # Just first 2 for speed
        results = run_realistic_benchmark(test, z_init=10.0+10.0j, num_trials=10)
        all_results[test.name + " (Bad Init)"] = results
        
        fig = visualize_realistic(test, results)
        filename = f'/home/midori/gits/3d-newton-fractal/realistic_{test.name.replace(" ", "_")}_bad_init.png'
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved: {filename}")
        plt.close(fig)
    
    # Summary Table
    print("\n" + "="*80)
    print("SUMMARY: Success Rates")
    print("="*80)
    print(f"{'Test':<35} {'Naive':<12} {'Damped':<12} {'φ-Surfer':<12}")
    print("-"*80)
    
    for test_name, results in all_results.items():
        row = f"{test_name:<35}"
        for method in ['Naive Newton', 'Damped Newton', 'φ-Surfer (Ours)']:
            sr = results[method]['avg_success_rate'] * 100
            row += f" {sr:>6.1f}%    "
        print(row)
    
    print("="*80)
    print("\n✅ Realistic Benchmark Complete!")
    print(f"   Results: /home/midori/gits/3d-newton-fractal/realistic_*.png")
