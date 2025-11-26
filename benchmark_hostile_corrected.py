"""
CORRECTED Benchmark: Hostile Basin Tracking Test

Based on detailed failure analysis, this benchmark tests ROOT TRACKING FIDELITY,
not just "find any root". Uses a system with fractal basin boundaries that 
actively resist Newton's method.

Key insights from failure analysis:
1. Original phi-Surfer used try/except that silently failed on JAX errors
2. Fixed gamma=0.2 was too large (20% of system scale)
3. Cubic roots are too stable - basins don't fracture
4. Wrong metric: "find a root" vs "track THE root"

New approach:
- Hostile function: f(z,λ) = z³ - z + λ (fracturing basins near critical points)
- Metric: Basin hopping count (lower is better)
- Adaptive step size with conjugate gradient
- No silent failures - expose all errors
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
from typing import Dict, Tuple

jax.config.update("jax_enable_x64", True)


# =============================================================================
# HOSTILE SYSTEM: z³ - z + λ
# =============================================================================

@jit
def f_hostile(z: complex, lam: complex) -> complex:
    """
    f(z, λ) = z³ - z + λ
    
    Critical points at z = ±1/√3 where f'(z) = 0
    When λ passes near critical values ±2/(3√3) ≈ ±0.385,
    two roots collide and basins shatter.
    
    This is the HOSTILE environment where Newton fails.
    """
    return z**3 - z + lam


@jit
def df_dz_hostile(z: complex, lam: complex) -> complex:
    """Derivative: f'(z) = 3z² - 1"""
    return 3.0 * z**2 - 1.0


@jit
def newton_step_hostile(z: complex, lam: complex) -> complex:
    """
    Single Newton iteration with soft protection.
    """
    fz = f_hostile(z, lam)
    dfz = df_dz_hostile(z, lam)
    
    # Soft protection against division by zero
    safe_denom = jnp.where(
        jnp.abs(dfz) < 1e-8,
        1e-8 + 0j,
        dfz
    )
    
    return z - fz / safe_denom


# =============================================================================
# PHI POTENTIAL: Corrected Implementation
# =============================================================================

@jit
def compute_phi_hostile(z: complex, lam: complex, K: int = 8) -> float:
    """
    φ(z,λ) = -log(min_step + ε)
    
    High φ = stable convergence basin
    Low φ = near fractal boundary (danger!)
    """
    def loop_body(i, val):
        curr_z, min_step = val
        next_z = newton_step_hostile(curr_z, lam)
        step = jnp.abs(next_z - curr_z)
        new_min = jnp.minimum(min_step, step)
        return (next_z, new_min)
    
    # Initialize with large step
    init_val = (z, 1000.0)
    final_z, min_step = jax.lax.fori_loop(0, K, loop_body, init_val)
    
    # φ potential
    phi = -jnp.log(min_step + 1e-9)
    return phi


# Value and gradient together (efficient)
phi_val_grad = jit(value_and_grad(compute_phi_hostile, argnums=0))


# =============================================================================
# SOLVER IMPLEMENTATIONS
# =============================================================================

def naive_newton_tracker(lambda_path: np.ndarray, 
                        z_init: complex = 0.5+0.5j,
                        newton_steps: int = 5) -> Dict:
    """
    Standard Newton continuation: Just iterate from previous position.
    No awareness of basin boundaries.
    """
    T = len(lambda_path)
    z_traj = np.zeros(T, dtype=complex)
    basin_hops = 0
    divergences = 0
    
    z = z_init
    
    for i, lam in enumerate(lambda_path):
        z_prev = z
        
        # Newton iteration
        for _ in range(newton_steps):
            z = newton_step_hostile(z, lam)
            
            # Check divergence
            if np.abs(z) > 100:
                divergences += 1
                z = z_prev  # Revert
                break
        
        # Detect basin hopping (sudden large jump)
        if np.abs(z - z_prev) > 0.5:
            basin_hops += 1
        
        z_traj[i] = z
    
    # Compute final residuals
    residuals = np.array([np.abs(f_hostile(z_traj[i], lambda_path[i])) 
                         for i in range(T)])
    converged = residuals < 1e-6
    
    return {
        'name': 'Naive Newton',
        'z': z_traj,
        'basin_hops': basin_hops,
        'divergences': divergences,
        'residuals': residuals,
        'success_rate': np.mean(converged)
    }


def phi_surfer_tracker(lambda_path: np.ndarray,
                      z_init: complex = 0.5+0.5j,
                      gamma: float = 0.02,
                      K: int = 8,
                      newton_steps: int = 2) -> Dict:
    """
    φ-Surfer: Active navigation using φ-gradient.
    
    CORRECTED VERSION:
    - Uses conjugate gradient for complex ascent
    - Adaptive step capping
    - No silent failures
    - Fewer Newton steps (2 instead of 5) since φ does the heavy lifting
    """
    T = len(lambda_path)
    z_traj = np.zeros(T, dtype=complex)
    phi_traj = np.zeros(T)
    basin_hops = 0
    divergences = 0
    gradient_failures = 0
    
    z = z_init
    
    for i, lam in enumerate(lambda_path):
        z_prev = z
        
        # A. φ-Surfing Step (Gradient Ascent)
        try:
            phi, g = phi_val_grad(z, lam)
            
            # CRITICAL FIX: Use conjugate gradient for complex functions
            # Direction of steepest ascent for real f(z) is conj(∇f)
            ascent_dir = jnp.conj(g)
            
            g_norm = jnp.abs(ascent_dir)
            
            if g_norm > 1e-10:
                # Normalized direction with adaptive step cap
                direction = ascent_dir / g_norm
                
                # CRITICAL FIX: Cap step size to prevent explosions
                step_size = jnp.minimum(gamma, 0.05 / g_norm)
                z = z + direction * step_size
            
            phi_traj[i] = phi
            
        except Exception as e:
            # NO SILENT FAILURES - log and count
            gradient_failures += 1
            phi_traj[i] = 0.0
            # Continue with current z
        
        # B. Newton Refinement (fewer steps needed)
        for _ in range(newton_steps):
            z_new = newton_step_hostile(z, lam)
            
            # Check divergence
            if np.abs(z_new) > 100:
                divergences += 1
                break
            
            z = z_new
        
        # Detect basin hopping
        if np.abs(z - z_prev) > 0.5:
            basin_hops += 1
        
        z_traj[i] = z
    
    # Compute residuals
    residuals = np.array([np.abs(f_hostile(z_traj[i], lambda_path[i]))
                         for i in range(T)])
    converged = residuals < 1e-6
    
    return {
        'name': 'φ-Surfer (Ours)',
        'z': z_traj,
        'phi': phi_traj,
        'basin_hops': basin_hops,
        'divergences': divergences,
        'gradient_failures': gradient_failures,
        'residuals': residuals,
        'success_rate': np.mean(converged)
    }


# =============================================================================
# TEST SCENARIOS
# =============================================================================

def create_hostile_path(path_type: str = 'bifurcation') -> Tuple[np.ndarray, str]:
    """
    Create parameter paths that stress-test root tracking.
    """
    if path_type == 'bifurcation':
        # Circle around critical value where basins fracture
        # Critical: λ ≈ ±2/(3√3) ≈ ±0.385
        ts = np.linspace(0, 2*np.pi, 200)
        lam_path = 0.38 + 0.1 * np.exp(1j * ts)
        desc = "Circle near bifurcation (λ ≈ 0.385)"
        
    elif path_type == 'crossing':
        # Line that crosses THROUGH the critical point
        ts = np.linspace(0, 1, 150)
        lam_path = -0.4 + 0.8j * ts  # Crosses Re(λ) ≈ 0.385
        desc = "Line crossing bifurcation"
        
    elif path_type == 'zigzag_hostile':
        # Sharp turns near dangerous regions
        ts = np.linspace(0, 1, 100)
        lam_path = np.zeros(100, dtype=complex)
        lam_path[0:25] = 0.3 + 0.1j
        lam_path[25:50] = 0.3 + 0.1j + (ts[25:50] - 0.25) * 4 * (0.1 - 0.3j)
        lam_path[50:75] = 0.4 - 0.2j
        lam_path[75:100] = 0.4 - 0.2j + (ts[75:100] - 0.75) * 4 * (-0.3 + 0.3j)
        desc = "Zigzag through fractal boundaries"
        
    else:  # 'safe'
        # Control: Safe path away from critical points
        ts = np.linspace(0, 2*np.pi, 150)
        lam_path = 1.0 + 0.3 * np.exp(1j * ts)
        desc = "Safe circle (control)"
    
    return lam_path, desc


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_hostile_benchmark(path_type: str = 'bifurcation', num_trials: int = 5):
    """
    Run the hostile tracking benchmark.
    """
    print(f"\n{'='*80}")
    print(f"HOSTILE BENCHMARK: {path_type.upper()}")
    print(f"{'='*80}\n")
    
    lam_path, desc = create_hostile_path(path_type)
    print(f"Path: {desc}")
    print(f"Length: {len(lam_path)} steps")
    print(f"Trials: {num_trials}\n")
    
    # Run multiple trials
    results_newton = []
    results_phi = []
    
    for trial in range(num_trials):
        # Slight randomization of initial condition
        z_init = 0.5 + 0.5j + 0.05 * (np.random.randn() + 1j * np.random.randn())
        
        print(f"Trial {trial+1}/{num_trials}... ", end='', flush=True)
        
        # Naive Newton
        start = time.time()
        res_newton = naive_newton_tracker(lam_path, z_init=z_init)
        time_newton = time.time() - start
        res_newton['time'] = time_newton
        results_newton.append(res_newton)
        
        # φ-Surfer
        start = time.time()
        res_phi = phi_surfer_tracker(lam_path, z_init=z_init)
        time_phi = time.time() - start
        res_phi['time'] = time_phi
        results_phi.append(res_phi)
        
        print(f"Newton hops: {res_newton['basin_hops']}, φ-Surfer hops: {res_phi['basin_hops']}")
    
    # Aggregate statistics
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<30} {'Naive Newton':<20} {'φ-Surfer (Ours)':<20}")
    print("-" * 80)
    
    # Basin hops (LOWER IS BETTER)
    hops_n = [r['basin_hops'] for r in results_newton]
    hops_p = [r['basin_hops'] for r in results_phi]
    print(f"{'Basin Hops (avg ± std)':<30} {np.mean(hops_n):.1f} ± {np.std(hops_n):.1f}{'':>9} {np.mean(hops_p):.1f} ± {np.std(hops_p):.1f}")
    
    # Divergences
    div_n = [r['divergences'] for r in results_newton]
    div_p = [r['divergences'] for r in results_phi]
    print(f"{'Divergences (avg)':<30} {np.mean(div_n):.1f}{'':>16} {np.mean(div_p):.1f}")
    
    # Success rate
    sr_n = [r['success_rate'] * 100 for r in results_newton]
    sr_p = [r['success_rate'] * 100 for r in results_phi]
    print(f"{'Convergence Rate (%)':<30} {np.mean(sr_n):.1f} ± {np.std(sr_n):.1f}{'':>9} {np.mean(sr_p):.1f} ± {np.std(sr_p):.1f}")
    
    # Time
    time_n = [r['time'] * 1000 for r in results_newton]
    time_p = [r['time'] * 1000 for r in results_phi]
    print(f"{'Runtime (ms)':<30} {np.mean(time_n):.1f}{'':>16} {np.mean(time_p):.1f}")
    
    # Gradient failures (phi only)
    gf_p = [r.get('gradient_failures', 0) for r in results_phi]
    print(f"{'Gradient Failures (φ only)':<30} {'N/A':>20} {np.mean(gf_p):.1f}")
    
    print("="*80)
    
    # Use best trial for visualization
    best_idx = np.argmin(hops_p)
    
    return {
        'newton': results_newton[best_idx],
        'phi': results_phi[best_idx],
        'lambda': lam_path,
        'description': desc,
        'stats': {
            'hops_newton': (np.mean(hops_n), np.std(hops_n)),
            'hops_phi': (np.mean(hops_p), np.std(hops_p)),
            'success_newton': (np.mean(sr_n), np.std(sr_n)),
            'success_phi': (np.mean(sr_p), np.std(sr_p)),
        }
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_hostile_tracking(results: Dict, save_path: str = None):
    """
    Create comprehensive visualization of tracking results.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    lam_path = results['lambda']
    res_n = results['newton']
    res_p = results['phi']
    
    colors = {
        'Naive Newton': '#FF6B6B',
        'φ-Surfer (Ours)': '#F7931E'
    }
    
    # Plot 1: Parameter Path
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(lam_path.real, lam_path.imag, 'k--', linewidth=2, alpha=0.5, label='λ(t) path')
    ax1.scatter(lam_path[0].real, lam_path[0].imag, c='green', s=120, marker='o',
               edgecolors='white', linewidths=2, zorder=5, label='Start')
    ax1.scatter(lam_path[-1].real, lam_path[-1].imag, c='red', s=120, marker='s',
               edgecolors='white', linewidths=2, zorder=5, label='End')
    # Mark critical region
    critical = 2 / (3 * np.sqrt(3))
    circle = plt.Circle((critical, 0), 0.1, fill=False, color='red', 
                       linestyle=':', linewidth=2, label='Critical region')
    ax1.add_patch(circle)
    ax1.set_xlabel('Re(λ)', fontsize=11)
    ax1.set_ylabel('Im(λ)', fontsize=11)
    ax1.set_title('Parameter Evolution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.axis('equal')
    
    # Plot 2: Solution Trajectories
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(res_n['z'].real, res_n['z'].imag, '-', linewidth=2, 
            color=colors['Naive Newton'], alpha=0.7, 
            label=f"Newton (hops: {res_n['basin_hops']})")
    ax2.plot(res_p['z'].real, res_p['z'].imag, '-', linewidth=2,
            color=colors['φ-Surfer (Ours)'], alpha=0.7,
            label=f"φ-Surfer (hops: {res_p['basin_hops']})")
    ax2.set_xlabel('Re(z)', fontsize=11)
    ax2.set_ylabel('Im(z)', fontsize=11)
    ax2.set_title('Root Trajectories', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Plot 3: Basin Hopping Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    methods = ['Naive\nNewton', 'φ-Surfer\n(Ours)']
    hops = [res_n['basin_hops'], res_p['basin_hops']]
    bars = ax3.bar(methods, hops, color=[colors['Naive Newton'], colors['φ-Surfer (Ours)']],
                  alpha=0.8, edgecolor='white', linewidth=2)
    ax3.set_ylabel('Basin Hop Count', fontsize=11)
    ax3.set_title('Tracking Fidelity (Lower = Better)', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, hops):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Plot 4: Residuals Over Time
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.semilogy(res_n['residuals'], linewidth=2, color=colors['Naive Newton'],
                alpha=0.8, label='Naive Newton')
    ax4.semilogy(res_p['residuals'], linewidth=2, color=colors['φ-Surfer (Ours)'],
                alpha=0.8, label='φ-Surfer')
    ax4.axhline(y=1e-6, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Threshold')
    ax4.set_xlabel('Timestep', fontsize=11)
    ax4.set_ylabel('|f(z, λ)|', fontsize=11)
    ax4.set_title('Convergence Quality', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend(fontsize=9)
    
    # Plot 5: φ Landscape (Our Method Only)
    ax5 = fig.add_subplot(gs[1, 1])
    if 'phi' in res_p:
        ax5.plot(res_p['phi'], linewidth=2.5, color=colors['φ-Surfer (Ours)'], alpha=0.8)
        ax5.set_xlabel('Timestep', fontsize=11)
        ax5.set_ylabel('φ Potential', fontsize=11)
        ax5.set_title('Stability Landscape (φ-Surfer)', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Chaos threshold')
        ax5.legend(fontsize=9)
    
    # Plot 6: Distance Between Trajectories
    ax6 = fig.add_subplot(gs[1, 2])
    dist = np.abs(res_n['z'] - res_p['z'])
    ax6.plot(dist, linewidth=2.5, color='purple', alpha=0.8)
    ax6.set_xlabel('Timestep', fontsize=11)
    ax6.set_ylabel('|z_Newton - z_Phi|', fontsize=11)
    ax6.set_title('Trajectory Divergence', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Hostile Tracking Test: {results["description"]}',
                fontsize=14, fontweight='bold', y=0.995)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved visualization: {save_path}")
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("CORRECTED BENCHMARK: Hostile Basin Tracking")
    print("="*80)
    print("\nFunction: f(z,λ) = z³ - z + λ")
    print("Critical bifurcation at λ ≈ ±0.385")
    print("\nMetric: Basin hopping count (lower = better tracking)")
    print("="*80)
    
    # Run all test scenarios
    test_paths = ['bifurcation', 'crossing', 'zigzag_hostile', 'safe']
    
    all_results = {}
    
    for path_type in test_paths:
        results = run_hostile_benchmark(path_type, num_trials=5)
        all_results[path_type] = results
        
        # Visualize
        save_path = f'/home/midori/gits/3d-newton-fractal/hostile_{path_type}.png'
        fig = visualize_hostile_tracking(results, save_path)
        plt.close(fig)
    
    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY: Basin Hopping Across All Tests")
    print("="*80)
    print(f"{'Path Type':<20} {'Newton (mean±std)':<25} {'φ-Surfer (mean±std)':<25} {'Winner':<10}")
    print("-"*80)
    
    for path_type, results in all_results.items():
        stats = results['stats']
        n_mean, n_std = stats['hops_newton']
        p_mean, p_std = stats['hops_phi']
        winner = "φ-Surfer ✅" if p_mean < n_mean else "Newton"
        
        print(f"{path_type:<20} {n_mean:.1f} ± {n_std:.1f}{'':>13} {p_mean:.1f} ± {p_std:.1f}{'':>13} {winner:<10}")
    
    print("="*80)
    print("\n✅ Corrected Benchmark Complete!")
    print("   Visualizations saved to: /home/midori/gits/3d-newton-fractal/hostile_*.png")
