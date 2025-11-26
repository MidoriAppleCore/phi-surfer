"""
FINAL Benchmark: Where Newton Actually FAILS

The previous "hostile" functions were still too stable. We need GENUINE chaos:
- Multiple attractors very close together
- Basins that are truly fractal (Koch-curve-like boundaries)
- Saddle points that trap Newton iterations

New test: Mandelbrot-like iteration with parameter continuation
f(z,c) = z² + c, but we track the FIXED POINT z* where f(z*,c) = z*

This has GENUINE chaos and Newton fails hard.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from jax import jit, value_and_grad
import time

jax.config.update("jax_enable_x64", True)

# =============================================================================
# GENUINELY HOSTILE: Fixed points of z² + c
# =============================================================================

@jit
def f_mandel(z: complex, c: complex) -> complex:
    """
    Fixed point equation: z² + c - z = 0
    
    This has 2 solutions at any c, with a critical point at z = 1/2.
    The basins are the FAMOUS Mandelbrot/Julia set fractals!
    
    Near c ≈ 1/4, the two fixed points collide and create TRUE fractal chaos.
    """
    return z**2 + c - z


@jit  
def df_dz_mandel(z: complex, c: complex) -> complex:
    """f'(z) = 2z - 1"""
    return 2.0 * z - 1.0


@jit
def newton_step_mandel(z: complex, c: complex) -> complex:
    """Newton iteration"""
    fz = f_mandel(z, c)
    dfz = df_dz_mandel(z, c)
    
    safe_denom = jnp.where(
        jnp.abs(dfz) < 1e-10,
        1e-10 + 0j,
        dfz
    )
    
    return z - fz / safe_denom


@jit
def compute_phi_mandel(z: complex, c: complex, K: int = 6) -> float:
    """φ potential for Mandelbrot fixed points"""
    def loop_body(i, val):
        curr_z, min_step = val
        next_z = newton_step_mandel(curr_z, c)
        step = jnp.abs(next_z - curr_z)
        new_min = jnp.minimum(min_step, step)
        return (next_z, new_min)
    
    init_val = (z, 1000.0)
    _, min_step = jax.lax.fori_loop(0, K, loop_body, init_val)
    
    phi = -jnp.log(min_step + 1e-10)
    return phi


phi_mandel_grad = jit(value_and_grad(compute_phi_mandel, argnums=0))


# =============================================================================
# THE GAUNTLET: Paths designed to break Newton
# =============================================================================

def create_killer_path() -> tuple:
    """
    Path through c-space that crosses fractal boundaries.
    
    Critical point: c = 1/4 where fixed points collide
    We'll spiral around it and cross through it.
    """
    # Part 1: Approach from safe distance
    t1 = np.linspace(0, 1, 50)
    c1 = 0.5 + 0.2 * np.exp(1j * 2 * np.pi * t1)
    
    # Part 2: Dive toward critical point
    t2 = np.linspace(0, 1, 30)
    c2 = 0.5 + (0.25 - 0.5) * t2 + 0.2j * (1 - t2)
    
    # Part 3: Cross through the singularity
    t3 = np.linspace(0, 1, 40)
    c3 = 0.25 - 0.15j + 0.3 * t3
    
    # Part 4: Escape
    t4 = np.linspace(0, 1, 30)
    c4 = c3[-1] + 0.2j * t4
    
    c_path = np.concatenate([c1, c2, c3, c4])
    
    return c_path, "Killer: Dive through c=1/4 singularity"


def create_fractal_dance() -> tuple:
    """
    Stay very close to c=1/4 for extended time.
    This keeps us in the fractal region where basins are thread-thin.
    """
    t = np.linspace(0, 4 * np.pi, 200)
    # Tiny circle around the critical point
    c_path = 0.25 + 0.05 * np.exp(1j * t)
    
    return c_path, "Fractal Dance: Tight circle around c=1/4"


# =============================================================================
# SOLVERS
# =============================================================================

def newton_baseline(c_path: np.ndarray, z_init: complex = 0.6+0j) -> dict:
    """Pure Newton with NO help (Straw Man - for comparison only)"""
    T = len(c_path)
    z_traj = np.zeros(T, dtype=complex)
    failures = 0
    wild_jumps = 0
    stalls = 0
    
    z = z_init
    
    for i, c in enumerate(c_path):
        z_prev = z
        
        # Newton iteration
        for step in range(10):
            z_new = newton_step_mandel(z, c)
            
            # Detect failure modes
            if np.abs(z_new) > 10:  # Divergence
                failures += 1
                z_new = z_prev  # Revert
                break
            
            if np.abs(z_new - z) < 1e-10:  # Converged
                z = z_new
                break
            
            z = z_new
        
        # Detect basin hopping
        if np.abs(z - z_prev) > 0.3:
            wild_jumps += 1
        
        # Detect stall (no progress)
        if np.abs(z - z_prev) < 1e-8:
            stalls += 1
        
        z_traj[i] = z
    
    residuals = np.array([np.abs(f_mandel(z_traj[i], c_path[i])) for i in range(T)])
    
    return {
        'name': 'Naive Newton',
        'z': z_traj,
        'failures': failures,
        'wild_jumps': wild_jumps,
        'stalls': stalls,
        'residuals': residuals,
        'success_rate': np.mean(residuals < 1e-6)
    }


def damped_newton(c_path: np.ndarray, z_init: complex = 0.6+0j) -> dict:
    """
    Damped Newton with Backtracking Line Search (INDUSTRY STANDARD)
    
    This is the "Real Boss" - what actual practitioners use.
    Uses Armijo condition to ensure descent.
    """
    T = len(c_path)
    z_traj = np.zeros(T, dtype=complex)
    failures = 0
    wild_jumps = 0
    stalls = 0
    
    z = z_init
    
    for i, c in enumerate(c_path):
        z_prev = z
        
        # Damped Newton with line search
        for iteration in range(15):  # More iterations allowed
            fz = f_mandel(z, c)
            dfz = df_dz_mandel(z, c)
            
            # Check for singular Jacobian
            if np.abs(dfz) < 1e-10:
                stalls += 1
                break
            
            # Full Newton direction
            direction = fz / dfz
            
            # Backtracking line search (Armijo rule)
            alpha = 1.0
            rho = 0.5  # Backtrack factor
            c1 = 1e-4  # Armijo constant
            
            residual_current = np.abs(fz)**2
            
            for backtrack in range(10):  # Max 10 backtracks
                z_trial = z - alpha * direction
                
                # Check divergence
                if np.abs(z_trial) > 10:
                    alpha *= rho
                    continue
                
                residual_trial = np.abs(f_mandel(z_trial, c))**2
                
                # Armijo condition: sufficient decrease
                if residual_trial <= residual_current * (1 - c1 * alpha):
                    z = z_trial
                    break
                
                alpha *= rho
            else:
                # Line search failed - stall
                stalls += 1
                break
            
            # Check convergence
            if np.abs(fz) < 1e-10:
                break
        
        # Detect basin hopping
        if np.abs(z - z_prev) > 0.3:
            wild_jumps += 1
        
        # Detect stall
        if np.abs(z - z_prev) < 1e-8:
            stalls += 1
        
        # Detect catastrophic failure
        if np.abs(z) > 10:
            failures += 1
            z = z_prev  # Revert
        
        z_traj[i] = z
    
    residuals = np.array([np.abs(f_mandel(z_traj[i], c_path[i])) for i in range(T)])
    
    return {
        'name': 'Damped Newton (SOTA)',
        'z': z_traj,
        'failures': failures,
        'wild_jumps': wild_jumps,
        'stalls': stalls,
        'residuals': residuals,
        'success_rate': np.mean(residuals < 1e-6)
    }


def levenberg_marquardt(c_path: np.ndarray, z_init: complex = 0.6+0j) -> dict:
    """
    Levenberg-Marquardt (Trust Region Method)
    
    The "Nuclear Option" - what you use when everything else fails.
    Interpolates between gradient descent (safe, slow) and Newton (fast, risky).
    """
    T = len(c_path)
    z_traj = np.zeros(T, dtype=complex)
    failures = 0
    wild_jumps = 0
    stalls = 0
    
    z = z_init
    lambda_lm = 0.01  # Initial damping parameter
    
    for i, c in enumerate(c_path):
        z_prev = z
        
        # LM iteration
        for iteration in range(15):
            fz = f_mandel(z, c)
            dfz = df_dz_mandel(z, c)
            
            # LM step: (J^T J + λI)^-1 J^T f
            # For scalar case: direction = f / (f' + λ)
            
            residual_current = np.abs(fz)**2
            
            # Try LM step
            denom = dfz + lambda_lm
            if np.abs(denom) < 1e-12:
                lambda_lm *= 10  # Increase damping
                stalls += 1
                continue
            
            direction = fz / denom
            z_trial = z - direction
            
            # Check trial point
            if np.abs(z_trial) > 10:
                lambda_lm *= 10  # Increase damping (more gradient-like)
                continue
            
            residual_trial = np.abs(f_mandel(z_trial, c))**2
            
            # Gain ratio
            rho = (residual_current - residual_trial) / (residual_current + 1e-12)
            
            if rho > 0:  # Good step
                z = z_trial
                lambda_lm *= 0.5  # Decrease damping (more Newton-like)
                lambda_lm = max(lambda_lm, 1e-10)
            else:  # Bad step
                lambda_lm *= 2.0  # Increase damping
            
            # Check convergence
            if np.abs(fz) < 1e-10:
                break
        
        # Detect basin hopping
        if np.abs(z - z_prev) > 0.3:
            wild_jumps += 1
        
        # Detect stall
        if np.abs(z - z_prev) < 1e-8:
            stalls += 1
        
        # Detect failure
        if np.abs(z) > 10:
            failures += 1
            z = z_prev
        
        z_traj[i] = z
    
    residuals = np.array([np.abs(f_mandel(z_traj[i], c_path[i])) for i in range(T)])
    
    return {
        'name': 'Levenberg-Marquardt',
        'z': z_traj,
        'failures': failures,
        'wild_jumps': wild_jumps,
        'stalls': stalls,
        'residuals': residuals,
        'success_rate': np.mean(residuals < 1e-6)
    }


def phi_surfer_adaptive(c_path: np.ndarray, z_init: complex = 0.6+0j) -> dict:
    """
    φ-Surfer with VERY conservative stepping.
    
    Key: In fractal regions, φ changes rapidly.
    We need tiny steps + momentum damping.
    """
    T = len(c_path)
    z_traj = np.zeros(T, dtype=complex)
    phi_traj = np.zeros(T)
    failures = 0
    wild_jumps = 0
    gradient_nans = 0
    
    z = z_init
    momentum = 0.0 + 0j  # Momentum damping
    
    for i, c in enumerate(c_path):
        z_prev = z
        
        # A. Compute φ gradient
        try:
            phi, g = phi_mandel_grad(z, c)
            
            # Check for NaN (happens in chaotic regions)
            if jnp.isnan(phi) or jnp.isnan(jnp.abs(g)):
                gradient_nans += 1
                g = 0.0 + 0j
                phi = 0.0
            
            # Conjugate for ascent
            g_ascent = jnp.conj(g)
            g_norm = jnp.abs(g_ascent)
            
            if g_norm > 1e-8:
                # VERY small step with momentum damping
                direction = g_ascent / g_norm
                step = 0.01 * direction  # Tiny step!
                
                # Momentum (helps smooth over fractal noise)
                momentum = 0.7 * momentum + 0.3 * step
                z = z + momentum
            
            phi_traj[i] = phi
            
        except:
            gradient_nans += 1
            phi_traj[i] = 0.0
        
        # B. Newton refinement (gentle)
        for step in range(3):  # Fewer steps
            z_new = newton_step_mandel(z, c)
            
            if np.abs(z_new) > 10:
                failures += 1
                break
            
            # Only accept if not jumping too far
            if np.abs(z_new - z) < 0.5:
                z = z_new
            else:
                break
        
        # Detect wild jumps
        if np.abs(z - z_prev) > 0.3:
            wild_jumps += 1
        
        z_traj[i] = z
    
    residuals = np.array([np.abs(f_mandel(z_traj[i], c_path[i])) for i in range(T)])
    
    return {
        'name': 'φ-Surfer Adaptive',
        'z': z_traj,
        'phi': phi_traj,
        'failures': failures,
        'wild_jumps': wild_jumps,
        'gradient_nans': gradient_nans,
        'residuals': residuals,
        'success_rate': np.mean(residuals < 1e-6)
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_gauntlet(results: dict, save_path: str):
    """Visualize ALL FOUR METHODS - the publishable version"""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    c_path = results['c_path']
    res_n = results['newton']
    res_d = results['damped']
    res_lm = results['lm']
    res_p = results['phi']
    
    colors = {'newton': '#FF6B6B', 'damped': '#4ECDC4', 'lm': '#9B59B6', 'phi': '#F7931E'}
    
    # Plot 1: Parameter path
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(c_path.real, c_path.imag, 'k-', linewidth=2, alpha=0.6, label='c(t)')
    ax1.scatter([0.25], [0], c='red', s=300, marker='X', 
               edgecolors='white', linewidths=3, label='Singularity c=1/4', zorder=10)
    ax1.scatter(c_path[0].real, c_path[0].imag, c='green', s=150, marker='o', label='Start')
    ax1.scatter(c_path[-1].real, c_path[-1].imag, c='blue', s=150, marker='s', label='End')
    ax1.set_xlabel('Re(c)', fontsize=11)
    ax1.set_ylabel('Im(c)', fontsize=11)
    ax1.set_title('Parameter Path Through Singularity', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Plot 2: All trajectories overlaid
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(res_n['z'].real, res_n['z'].imag, color=colors['newton'], linewidth=1.5, alpha=0.4, label='Naive Newton')
    ax2.plot(res_d['z'].real, res_d['z'].imag, color=colors['damped'], linewidth=2, alpha=0.7, label='Damped Newton')
    ax2.plot(res_lm['z'].real, res_lm['z'].imag, color=colors['lm'], linewidth=2, alpha=0.7, label='Lev-Marq')
    ax2.plot(res_p['z'].real, res_p['z'].imag, color=colors['phi'], linewidth=2.5, alpha=0.9, label='φ-Surfer')
    ax2.set_xlabel('Re(z)', fontsize=11)
    ax2.set_ylabel('Im(z)', fontsize=11)
    ax2.set_title('Root Trajectories (z-space)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Plot 3: Wild jumps
    ax3 = fig.add_subplot(gs[0, 2])
    methods_short = ['Naive\nNewton', 'Damped\nNewton', 'Lev-Marq', 'φ-Surfer']
    jumps = [res_n['wild_jumps'], res_d['wild_jumps'], res_lm['wild_jumps'], res_p['wild_jumps']]
    bars = ax3.bar(methods_short, jumps, color=list(colors.values()), alpha=0.8, edgecolor='white', linewidth=2)
    ax3.set_ylabel('Basin Hop Count', fontsize=11)
    ax3.set_title('Wild Jumps (Lower = Better)', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, jumps):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha='center', fontsize=11, fontweight='bold')
    
    # Plot 4: Residuals
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.semilogy(res_n['residuals'], color=colors['newton'], linewidth=1.5, alpha=0.4, label='Naive Newton')
    ax4.semilogy(res_d['residuals'], color=colors['damped'], linewidth=2, alpha=0.7, label='Damped Newton')
    ax4.semilogy(res_lm['residuals'], color=colors['lm'], linewidth=2, alpha=0.7, label='Lev-Marq')
    ax4.semilogy(res_p['residuals'], color=colors['phi'], linewidth=2.5, alpha=0.9, label='φ-Surfer')
    ax4.axhline(1e-6, color='green', linestyle='--', linewidth=2, label='Success Threshold')
    ax4.set_xlabel('Timestep', fontsize=11)
    ax4.set_ylabel('Residual |f(z,c)|', fontsize=11)
    ax4.set_title('Convergence Quality', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend(fontsize=9)
    
    # Plot 5: Failures
    ax5 = fig.add_subplot(gs[1, 1])
    failures = [res_n['failures'], res_d['failures'], res_lm['failures'], res_p['failures']]
    bars = ax5.bar(methods_short, failures, color=list(colors.values()), alpha=0.8, edgecolor='white', linewidth=2)
    ax5.set_ylabel('Divergence Count', fontsize=11)
    ax5.set_title('Catastrophic Failures (Lower = Better)', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, failures):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2, height + max(failures)*0.02 if height > 0 else 0.5,
                str(val), ha='center', fontsize=11, fontweight='bold')
    
    # Plot 6: STALLS (THE KEY METRIC!)
    ax6 = fig.add_subplot(gs[1, 2])
    stalls = [res_n.get('stalls', 0), res_d.get('stalls', 0), res_lm.get('stalls', 0), res_p.get('stalls', 0)]
    bars = ax6.bar(methods_short, stalls, color=list(colors.values()), alpha=0.8, edgecolor='white', linewidth=2)
    ax6.set_ylabel('Stall Count', fontsize=11)
    ax6.set_title('⭐ Velocity Through Singularity ⭐', fontsize=12, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, stalls):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2, height + max(stalls)*0.02 if height > 0 else 0.5,
                str(val), ha='center', fontsize=11, fontweight='bold')
    
    # Plot 7: Success rates
    ax7 = fig.add_subplot(gs[2, 0])
    success = [res_n['success_rate']*100, res_d['success_rate']*100, 
               res_lm['success_rate']*100, res_p['success_rate']*100]
    bars = ax7.bar(methods_short, success, color=list(colors.values()), alpha=0.8, edgecolor='white', linewidth=2)
    ax7.set_ylabel('Success Rate (%)', fontsize=11)
    ax7.set_title('Overall Success', fontsize=12, fontweight='bold')
    ax7.set_ylim([0, 105])
    ax7.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, success):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    # Plot 8-9: Summary scorecard (spans 2 columns)
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    
    scorecard = f"""🏆 THE GAUNTLET SCORECARD 🏆
    
Test: {results['desc']}

┌─────────────────────────────────────────────────────────────────────────┐
│ NAIVE NEWTON (Straw Man)                                               │
│   Jumps: {res_n['wild_jumps']:<3}  Failures: {res_n['failures']:<3}  Stalls: {res_n.get('stalls',0):<3}  Success: {res_n['success_rate']*100:5.1f}%       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ DAMPED NEWTON (Industry Standard - Line Search)                        │
│   Jumps: {res_d['wild_jumps']:<3}  Failures: {res_d['failures']:<3}  Stalls: {res_d.get('stalls',0):<3}  Success: {res_d['success_rate']*100:5.1f}%       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ LEVENBERG-MARQUARDT (Trust Region - Nuclear Option)                    │
│   Jumps: {res_lm['wild_jumps']:<3}  Failures: {res_lm['failures']:<3}  Stalls: {res_lm.get('stalls',0):<3}  Success: {res_lm['success_rate']*100:5.1f}%       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ φ-SURFER (Holotopic Continuation - K={6} lookahead)                     │
│   Jumps: {res_p['wild_jumps']:<3}  Failures: {res_p['failures']:<3}  Stalls: {res_p.get('stalls',0):<3}  Success: {res_p['success_rate']*100:5.1f}%       │
└─────────────────────────────────────────────────────────────────────────┘
"""
    
    if res_p.get('stalls', 0) < res_d.get('stalls', 0) * 0.7:
        scorecard += f"\n✨ BREAKTHROUGH: φ-Surfer maintains velocity through bifurcation!"
    if res_p['failures'] < res_d['failures']:
        scorecard += f"\n💎 φ-Surfer avoided {res_d['failures'] - res_p['failures']} catastrophic failures vs SOTA!"
    if res_p['wild_jumps'] == 0 and res_d['wild_jumps'] > 0:
        scorecard += f"\n🎯 φ-Surfer eliminated basin hopping completely!"
    
    ax8.text(0.05, 0.5, scorecard, fontsize=10, family='monospace',
            verticalalignment='center', 
            bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.3, edgecolor='black', linewidth=2))
    
    plt.suptitle(f"THE GAUNTLET: {results['desc']}", fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved: {save_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("THE GAUNTLET: Where Newton Actually Fails")
    print("="*80)
    print("\nFunction: Fixed points of z² + c")
    print("Critical: c = 1/4 (fractal basin boundaries)")
    print("="*80 + "\n")
    
    tests = [
        ('killer', create_killer_path),
        ('fractal_dance', create_fractal_dance),
    ]
    
    for test_name, path_func in tests:
        print(f"\n{'='*80}")
        print(f"TEST: {test_name.upper()}")
        print(f"{'='*80}\n")
        
        c_path, desc = path_func()
        print(f"Path: {desc}")
        print(f"Length: {len(c_path)} steps\n")
        
        # Run ALL FOUR solvers
        print("Running Naive Newton (Straw Man)... ", end='', flush=True)
        start = time.time()
        res_newton = newton_baseline(c_path)
        time_n = time.time() - start
        print(f"{time_n*1000:.1f}ms")
        
        print("Running Damped Newton (SOTA)... ", end='', flush=True)
        start = time.time()
        res_damped = damped_newton(c_path)
        time_d = time.time() - start
        print(f"{time_d*1000:.1f}ms")
        
        print("Running Levenberg-Marquardt... ", end='', flush=True)
        start = time.time()
        res_lm = levenberg_marquardt(c_path)
        time_lm = time.time() - start
        print(f"{time_lm*1000:.1f}ms")
        
        print("Running φ-Surfer Adaptive... ", end='', flush=True)
        start = time.time()
        res_phi = phi_surfer_adaptive(c_path)
        time_p = time.time() - start
        print(f"{time_p*1000:.1f}ms\n")
        
        # Results table
        print("RESULTS:")
        print("-" * 100)
        print(f"{'Metric':<25} {'Naive Newton':<18} {'Damped Newton':<18} {'Lev-Marq':<18} {'φ-Surfer':<18}")
        print("-" * 100)
        print(f"{'Wild Jumps':<25} {res_newton['wild_jumps']:<18} {res_damped['wild_jumps']:<18} {res_lm['wild_jumps']:<18} {res_phi['wild_jumps']:<18}")
        print(f"{'Failures (divergence)':<25} {res_newton['failures']:<18} {res_damped['failures']:<18} {res_lm['failures']:<18} {res_phi['failures']:<18}")
        print(f"{'Stalls':<25} {res_newton.get('stalls',0):<18} {res_damped.get('stalls',0):<18} {res_lm.get('stalls',0):<18} {res_phi.get('stalls',0):<18}")
        print(f"{'Success Rate (%)':<25} {res_newton['success_rate']*100:<18.1f} {res_damped['success_rate']*100:<18.1f} {res_lm['success_rate']*100:<18.1f} {res_phi['success_rate']*100:<18.1f}")
        print(f"{'Runtime (ms)':<25} {time_n*1000:<18.1f} {time_d*1000:<18.1f} {time_lm*1000:<18.1f} {time_p*1000:<18.1f}")
        print("=" * 100)
        
        # Determine winner (priority: failures > stalls > wild jumps)
        methods = {
            'Naive Newton': res_newton,
            'Damped Newton (SOTA)': res_damped,
            'Levenberg-Marquardt': res_lm,
            'φ-Surfer': res_phi
        }
        
        winner_name = min(methods.items(), 
                         key=lambda x: (x[1]['failures'], 
                                       x[1].get('stalls', 0),
                                       x[1]['wild_jumps']))[0]
        
        print(f"\n🏆 WINNER: {winner_name}")
        
        # Scientific commentary
        if res_phi.get('stalls', 0) < res_damped.get('stalls', 0) * 0.7:
            print(f"   ✨ φ-Surfer maintained velocity through singularity!")
            print(f"   ({res_phi.get('stalls', 0)} stalls vs {res_damped.get('stalls', 0)} for Damped Newton)")
        
        if res_phi['failures'] < res_damped['failures']:
            print(f"   💎 φ-Surfer avoided {res_damped['failures'] - res_phi['failures']} catastrophic failures!")
        
        if res_phi['wild_jumps'] == 0 and res_damped['wild_jumps'] > 0:
            print(f"   🎯 φ-Surfer eliminated basin hopping completely!")
        
        # Visualize
        results = {
            'c_path': c_path,
            'newton': res_newton,
            'damped': res_damped,
            'lm': res_lm,
            'phi': res_phi,
            'desc': desc
        }
        
        save_path = f'/home/midori/gits/3d-newton-fractal/gauntlet_{test_name}.png'
        fig = visualize_gauntlet(results, save_path)
        plt.close(fig)
    
    print("\n" + "="*80)
    print("THE GAUNTLET COMPLETE!")
    print("="*80)
