#!/usr/bin/env python3
"""
THE ROBOT GAUNTLET: φ-Surfer for Inverse Kinematics
====================================================

Test φ-gradient navigation on a REAL robotics problem:
- 2-link planar robot arm
- Kinematic singularity when fully extended
- Task: Move end-effector through/near singularity

This is the "Robot Translation" of the Mandelbrot benchmark.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
from typing import Tuple, Dict
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

print("="*80)
print("THE ROBOT GAUNTLET: Inverse Kinematics Through Singularities")
print("="*80)
print(f"JAX version: {jax.__version__}")
print(f"Device: {jax.devices()[0]}")
print("="*80 + "\n")


# =============================================================================
# ROBOT ARM KINEMATICS (2-link planar manipulator)
# =============================================================================

L1 = 1.0  # Link 1 length
L2 = 1.0  # Link 2 length

@jit
def forward_kinematics(theta: jnp.ndarray) -> jnp.ndarray:
    """
    Forward kinematics: joint angles → end-effector position
    
    Args:
        theta: [theta1, theta2] joint angles (radians)
    
    Returns:
        x: [x, y] end-effector position
    """
    theta1, theta2 = theta
    x = L1 * jnp.cos(theta1) + L2 * jnp.cos(theta1 + theta2)
    y = L1 * jnp.sin(theta1) + L2 * jnp.sin(theta1 + theta2)
    return jnp.array([x, y])


@jit
def inverse_kinematics_residual(theta: jnp.ndarray, x_target: jnp.ndarray) -> jnp.ndarray:
    """
    IK residual: f(θ, x_target) = FK(θ) - x_target
    
    We want to solve: FK(θ) = x_target
    """
    x_current = forward_kinematics(theta)
    return x_current - x_target


@jit
def jacobian(theta: jnp.ndarray) -> jnp.ndarray:
    """
    Jacobian matrix J = ∂FK/∂θ
    
    Singularity occurs when det(J) = 0 (robot fully extended or folded)
    """
    theta1, theta2 = theta
    
    # Analytical Jacobian for 2-link planar arm
    J = jnp.array([
        [-L1*jnp.sin(theta1) - L2*jnp.sin(theta1+theta2), -L2*jnp.sin(theta1+theta2)],
        [ L1*jnp.cos(theta1) + L2*jnp.cos(theta1+theta2),  L2*jnp.cos(theta1+theta2)]
    ])
    return J


@jit
def manipulability(theta: jnp.ndarray) -> float:
    """
    Yoshikawa manipulability index: sqrt(det(J*J^T))
    
    Goes to ZERO at singularities!
    """
    J = jacobian(theta)
    return jnp.sqrt(jnp.linalg.det(J @ J.T))


@jit
def newton_step_ik(theta: jnp.ndarray, x_target: jnp.ndarray) -> jnp.ndarray:
    """One step of Newton-Raphson for IK: θ_new = θ - J^{-1} * f"""
    J = jacobian(theta)
    f = inverse_kinematics_residual(theta, x_target)
    
    # Pseudo-inverse for safety
    J_pinv = jnp.linalg.pinv(J, rcond=1e-8)
    delta_theta = J_pinv @ f
    
    return theta - delta_theta


# =============================================================================
# φ-POTENTIAL FOR ROBOTICS (SIMPLIFIED - NO LOOKAHEAD)
# =============================================================================

@jit
def phi_potential_ik_simple(theta: jnp.ndarray) -> float:
    """
    SIMPLIFIED φ-potential: Uses CURRENT manipulability (no unstable lookahead)
    
    φ(θ) = 1 / (m(θ)² + λ)
    
    High φ → low manipulability → near singularity
    
    This avoids the unstable nested Newton iterations!
    """
    m = manipulability(theta)
    lambda_reg = 0.01
    
    # Quadratic inverse to make gradient stronger
    phi = 1.0 / (m**2 + lambda_reg)
    
    return phi


# Gradient of SIMPLE φ
phi_grad_theta_simple_jit = jit(grad(phi_potential_ik_simple))


# =============================================================================
# TEST TRAJECTORIES (End-Effector Paths)
# =============================================================================

def create_singularity_crossing() -> Tuple[np.ndarray, str]:
    """
    Path that crosses through the singularity (fully extended)
    
    This is the "Killer" test for robots.
    """
    T = 150
    t = np.linspace(0, 1, T)
    
    # Start at (1.5, 0.5), go to (1.5, -0.5)
    # This crosses the singularity circle at radius = L1 + L2 = 2.0
    x_path = np.zeros((T, 2))
    x_path[:, 0] = 1.5 + 0.5 * np.sin(2*np.pi*t)  # Crosses r=2 twice
    x_path[:, 1] = 0.5 * np.cos(2*np.pi*t)
    
    return x_path, "Crossing through singularity (r=2.0)"


def create_singularity_dance() -> Tuple[np.ndarray, str]:
    """
    Path that circles near (but not through) the singularity
    
    This is the "Fractal Dance" test for robots.
    """
    T = 200
    theta_traj = np.linspace(0, 2*np.pi, T)
    
    # Circle at radius 1.9 (just inside singularity at r=2.0)
    r = 1.9
    x_path = np.zeros((T, 2))
    x_path[:, 0] = r * np.cos(theta_traj)
    x_path[:, 1] = r * np.sin(theta_traj)
    
    return x_path, "Dancing around singularity (r=1.9, singularity at r=2.0)"


# =============================================================================
# SOLVERS
# =============================================================================

def naive_newton_ik(x_path: np.ndarray, theta_init: np.ndarray = None) -> dict:
    """Naive Newton-Raphson IK (Straw Man)"""
    if theta_init is None:
        theta_init = np.array([np.pi/4, np.pi/4])
    
    T = len(x_path)
    theta_traj = np.zeros((T, 2))
    failures = 0
    wild_jumps = 0
    stalls = 0
    
    theta = theta_init
    
    for i, x_target in enumerate(x_path):
        theta_prev = theta.copy()
        
        # Newton iterations
        for iteration in range(10):
            theta_new = np.array(newton_step_ik(theta, x_target))
            
            # Check for NaN or huge values
            if np.any(np.isnan(theta_new)) or np.linalg.norm(theta_new) > 10:
                failures += 1
                theta_new = theta_prev
                break
            
            # Check convergence
            if np.linalg.norm(theta_new - theta) < 1e-10:
                theta = theta_new
                break
            
            theta = theta_new
        
        # Detect wild jump (large joint change)
        if np.linalg.norm(theta - theta_prev) > 0.5:
            wild_jumps += 1
        
        # Detect stall
        if np.linalg.norm(theta - theta_prev) < 1e-8:
            stalls += 1
        
        theta_traj[i] = theta
    
    # Compute residuals
    residuals = np.array([np.linalg.norm(forward_kinematics(theta_traj[i]) - x_path[i]) 
                          for i in range(T)])
    
    # Compute manipulability along trajectory
    manip = np.array([float(manipulability(theta_traj[i])) for i in range(T)])
    
    return {
        'name': 'Naive Newton',
        'theta': theta_traj,
        'failures': failures,
        'wild_jumps': wild_jumps,
        'stalls': stalls,
        'residuals': residuals,
        'manipulability': manip,
        'success_rate': np.mean(residuals < 1e-4)
    }


def damped_newton_ik(x_path: np.ndarray, theta_init: np.ndarray = None) -> dict:
    """Damped Newton with Backtracking Line Search (SOTA for robotics)"""
    if theta_init is None:
        theta_init = np.array([np.pi/4, np.pi/4])
    
    T = len(x_path)
    theta_traj = np.zeros((T, 2))
    failures = 0
    wild_jumps = 0
    stalls = 0
    
    theta = theta_init
    
    for i, x_target in enumerate(x_path):
        theta_prev = theta.copy()
        
        for iteration in range(15):
            J = np.array(jacobian(theta))
            f = np.array(inverse_kinematics_residual(theta, x_target))
            
            # Check singularity
            if np.linalg.cond(J) > 1e6:
                stalls += 1
                break
            
            # Pseudo-inverse step
            J_pinv = np.linalg.pinv(J, rcond=1e-8)
            direction = J_pinv @ f
            
            # Backtracking line search
            alpha = 1.0
            residual_current = np.linalg.norm(f)**2
            
            for backtrack in range(10):
                theta_trial = theta - alpha * direction
                residual_trial = np.linalg.norm(inverse_kinematics_residual(theta_trial, x_target))**2
                
                # Armijo condition
                if residual_trial <= residual_current * (1 - 1e-4 * alpha):
                    theta = theta_trial
                    break
                
                alpha *= 0.5
            else:
                stalls += 1
                break
            
            if np.linalg.norm(f) < 1e-10:
                break
        
        # Detect jumps and stalls
        if np.linalg.norm(theta - theta_prev) > 0.5:
            wild_jumps += 1
        if np.linalg.norm(theta - theta_prev) < 1e-8:
            stalls += 1
        
        theta_traj[i] = theta
    
    residuals = np.array([np.linalg.norm(forward_kinematics(theta_traj[i]) - x_path[i]) 
                          for i in range(T)])
    manip = np.array([float(manipulability(theta_traj[i])) for i in range(T)])
    
    return {
        'name': 'Damped Newton (SOTA)',
        'theta': theta_traj,
        'failures': failures,
        'wild_jumps': wild_jumps,
        'stalls': stalls,
        'residuals': residuals,
        'manipulability': manip,
        'success_rate': np.mean(residuals < 1e-4)
    }


def phi_surfer_ik(x_path: np.ndarray, theta_init: np.ndarray = None) -> dict:
    """
    φ-Surfer for Inverse Kinematics (SIMPLIFIED VERSION)
    
    Uses CURRENT manipulability gradient to steer away from singularities.
    No unstable lookahead needed!
    
    Strategy:
    1. Compute ∇φ (gradient of inverse manipulability)
    2. Step in NEGATIVE gradient direction (toward higher manipulability = safer)
    3. Refine with damped Newton to track target
    """
    if theta_init is None:
        theta_init = np.array([np.pi/4, np.pi/4])
    
    T = len(x_path)
    theta_traj = np.zeros((T, 2))
    failures = 0
    wild_jumps = 0
    stalls = 0
    phi_history = []
    gradient_nans = 0
    
    theta = theta_init
    
    for i, x_target in enumerate(x_path):
        theta_prev = theta.copy()
        
        # Compute φ-gradient (singularity avoidance using CURRENT state)
        try:
            g_phi = np.array(phi_grad_theta_simple_jit(theta))
            
            if np.any(np.isnan(g_phi)) or np.any(np.isinf(g_phi)):
                gradient_nans += 1
                g_phi = np.zeros(2)  # No steering
            
            # Also compute task-space gradient (toward target)
            J = np.array(jacobian(theta))
            f = np.array(inverse_kinematics_residual(theta, x_target))
            J_pinv = np.linalg.pinv(J, rcond=1e-6)
            g_task = J_pinv @ f  # Direction toward target
            
            # BLEND: singularity avoidance + task tracking
            g_phi_norm = np.linalg.norm(g_phi)
            
            # When close to singularity, prioritize avoidance
            m_current = float(manipulability(theta))
            if m_current < 0.2:  # Close to singularity!
                weight_avoid = 0.7
                weight_task = 0.3
            else:  # Safe region
                weight_avoid = 0.2
                weight_task = 0.8
            
            if g_phi_norm > 1e-8:
                # Descend φ-gradient (move toward higher manipulability)
                step_avoid = -0.05 * g_phi / g_phi_norm
            else:
                step_avoid = np.zeros(2)
            
            # Task step
            step_task = -0.5 * g_task
            
            # Blended step
            theta = theta + weight_avoid * step_avoid + weight_task * step_task
            
            # Refine with damped Newton (converge to target)
            for refinement in range(5):
                J = np.array(jacobian(theta))
                f = np.array(inverse_kinematics_residual(theta, x_target))
                
                if np.linalg.cond(J) > 1e6:
                    # Near singularity - very small step
                    J_pinv = np.linalg.pinv(J, rcond=1e-4)
                    theta = theta - 0.1 * J_pinv @ f
                else:
                    J_pinv = np.linalg.pinv(J, rcond=1e-8)
                    theta = theta - J_pinv @ f
                
                if np.linalg.norm(f) < 1e-8:
                    break
            
            # Record φ
            phi_val = float(phi_potential_ik_simple(theta))
            phi_history.append(phi_val)
            
        except Exception as e:
            failures += 1
            # Emergency fallback
            try:
                J = np.array(jacobian(theta_prev))
                f = np.array(inverse_kinematics_residual(theta_prev, x_target))
                J_pinv = np.linalg.pinv(J, rcond=1e-4)
                theta = theta_prev - 0.1 * J_pinv @ f
            except:
                theta = theta_prev  # Last resort: don't move
        
        # Detect jumps and stalls
        delta = np.linalg.norm(theta - theta_prev)
        if delta > 0.5:
            wild_jumps += 1
        if delta < 1e-8:
            stalls += 1
        
        theta_traj[i] = theta
    
    residuals = np.array([np.linalg.norm(forward_kinematics(theta_traj[i]) - x_path[i]) 
                          for i in range(T)])
    manip = np.array([float(manipulability(theta_traj[i])) for i in range(T)])
    
    return {
        'name': 'φ-Surfer',
        'theta': theta_traj,
        'failures': failures,
        'wild_jumps': wild_jumps,
        'stalls': stalls,
        'residuals': residuals,
        'manipulability': manip,
        'phi': np.array(phi_history) if phi_history else np.zeros(T),
        'gradient_nans': gradient_nans,
        'success_rate': np.mean(residuals < 1e-4)
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_robot_results(results: dict, save_path: str):
    """Visualize robot IK results"""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    x_path = results['x_path']
    res_n = results['newton']
    res_d = results['damped']
    res_p = results['phi']
    
    colors = {'newton': '#FF6B6B', 'damped': '#4ECDC4', 'phi': '#F7931E'}
    
    # Plot 1: End-effector path
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x_path[:, 0], x_path[:, 1], 'k--', linewidth=2, alpha=0.6, label='Desired Path')
    
    # Draw singularity circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    r_sing = L1 + L2
    ax1.plot(r_sing * np.cos(theta_circle), r_sing * np.sin(theta_circle), 
             'r-', linewidth=3, alpha=0.3, label='Singularity (r=2.0)')
    
    ax1.scatter(x_path[0, 0], x_path[0, 1], c='green', s=150, marker='o', label='Start', zorder=10)
    ax1.scatter(x_path[-1, 0], x_path[-1, 1], c='blue', s=150, marker='s', label='End', zorder=10)
    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Y (m)', fontsize=11)
    ax1.set_title('End-Effector Target Path', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend(fontsize=9)
    
    # Plot 2: Joint trajectories
    ax2 = fig.add_subplot(gs[0, 1])
    T = len(x_path)
    t = np.arange(T)
    ax2.plot(t, res_n['theta'][:, 0], color=colors['newton'], linewidth=1.5, alpha=0.4, label='Naive θ1')
    ax2.plot(t, res_d['theta'][:, 0], color=colors['damped'], linewidth=2, alpha=0.7, label='Damped θ1')
    ax2.plot(t, res_p['theta'][:, 0], color=colors['phi'], linewidth=2.5, alpha=0.9, label='φ-Surfer θ1')
    ax2.set_xlabel('Time Step', fontsize=11)
    ax2.set_ylabel('θ1 (rad)', fontsize=11)
    ax2.set_title('Joint 1 Trajectory', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Plot 3: Manipulability (SINGULARITY METRIC!)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, res_n['manipulability'], color=colors['newton'], linewidth=1.5, alpha=0.4, label='Naive Newton')
    ax3.plot(t, res_d['manipulability'], color=colors['damped'], linewidth=2, alpha=0.7, label='Damped Newton')
    ax3.plot(t, res_p['manipulability'], color=colors['phi'], linewidth=2.5, alpha=0.9, label='φ-Surfer')
    ax3.axhline(0.1, color='red', linestyle='--', linewidth=2, label='Danger Zone', alpha=0.5)
    ax3.set_xlabel('Time Step', fontsize=11)
    ax3.set_ylabel('Manipulability', fontsize=11)
    ax3.set_title('⭐ Distance from Singularity ⭐', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    # Plot 4: Tracking error
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.semilogy(t, res_n['residuals'], color=colors['newton'], linewidth=1.5, alpha=0.4, label='Naive Newton')
    ax4.semilogy(t, res_d['residuals'], color=colors['damped'], linewidth=2, alpha=0.7, label='Damped Newton')
    ax4.semilogy(t, res_p['residuals'], color=colors['phi'], linewidth=2.5, alpha=0.9, label='φ-Surfer')
    ax4.axhline(1e-4, color='green', linestyle='--', linewidth=2, label='Success Threshold')
    ax4.set_xlabel('Time Step', fontsize=11)
    ax4.set_ylabel('Tracking Error (m)', fontsize=11)
    ax4.set_title('End-Effector Accuracy', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend(fontsize=9)
    
    # Plot 5: Stalls comparison
    ax5 = fig.add_subplot(gs[1, 1])
    methods = ['Naive\nNewton', 'Damped\nNewton', 'φ-Surfer']
    stalls = [res_n['stalls'], res_d['stalls'], res_p['stalls']]
    bars = ax5.bar(methods, stalls, color=list(colors.values()), alpha=0.8, edgecolor='white', linewidth=2)
    ax5.set_ylabel('Stall Count', fontsize=11)
    ax5.set_title('Robot Freezes Near Singularity', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, stalls):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2, height + max(stalls)*0.02 if height > 0 else 0.5,
                str(val), ha='center', fontsize=11, fontweight='bold')
    
    # Plot 6: Wild jumps
    ax6 = fig.add_subplot(gs[1, 2])
    jumps = [res_n['wild_jumps'], res_d['wild_jumps'], res_p['wild_jumps']]
    bars = ax6.bar(methods, jumps, color=list(colors.values()), alpha=0.8, edgecolor='white', linewidth=2)
    ax6.set_ylabel('Wild Jump Count', fontsize=11)
    ax6.set_title('Joint Discontinuities', fontsize=12, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, jumps):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha='center', fontsize=11, fontweight='bold')
    
    # Plot 7: Success rates
    ax7 = fig.add_subplot(gs[2, 0])
    success = [res_n['success_rate']*100, res_d['success_rate']*100, res_p['success_rate']*100]
    bars = ax7.bar(methods, success, color=list(colors.values()), alpha=0.8, edgecolor='white', linewidth=2)
    ax7.set_ylabel('Success Rate (%)', fontsize=11)
    ax7.set_title('Task Completion Rate', fontsize=12, fontweight='bold')
    ax7.set_ylim([0, 105])
    ax7.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, success):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    # Plot 8-9: Scorecard
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    
    scorecard = f"""THE ROBOT GAUNTLET SCORECARD
    
Task: {results['desc']}
Robot: 2-link planar arm (L1={L1}m, L2={L2}m)
Singularity: Fully extended (r=2.0m)

NAIVE NEWTON (Straw Man):
  Stalls: {res_n['stalls']:<3}  Wild Jumps: {res_n['wild_jumps']:<3}  Failures: {res_n['failures']:<3}  Success: {res_n['success_rate']*100:5.1f}%

DAMPED NEWTON (Industry Standard):
  Stalls: {res_d['stalls']:<3}  Wild Jumps: {res_d['wild_jumps']:<3}  Failures: {res_d['failures']:<3}  Success: {res_d['success_rate']*100:5.1f}%

PHI-SURFER (Holotopic Navigation):
  Stalls: {res_p['stalls']:<3}  Wild Jumps: {res_p['wild_jumps']:<3}  Failures: {res_p['failures']:<3}  Success: {res_p['success_rate']*100:5.1f}%
"""
    
    if res_p['stalls'] < res_d['stalls'] * 0.7:
        scorecard += f"\n✨ φ-Surfer navigated smoothly through singularity!"
    if res_p['wild_jumps'] == 0 and res_d['wild_jumps'] > 0:
        scorecard += f"\n🎯 φ-Surfer eliminated joint discontinuities!"
    
    ax8.text(0.05, 0.5, scorecard, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.3, edgecolor='black', linewidth=2))
    
    plt.suptitle(f"ROBOT IK: {results['desc']}", fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Robot visualization saved: {save_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\nRunning robot IK benchmarks...\n")
    
    tests = [
        ('singularity_crossing', create_singularity_crossing),
        ('singularity_dance', create_singularity_dance),
    ]
    
    for test_name, path_func in tests:
        print(f"\n{'='*80}")
        print(f"TEST: {test_name.upper().replace('_', ' ')}")
        print(f"{'='*80}\n")
        
        x_path, desc = path_func()
        print(f"Path: {desc}")
        print(f"Length: {len(x_path)} waypoints\n")
        
        # Run solvers
        print("Running Naive Newton IK... ", end='', flush=True)
        start = time.time()
        res_newton = naive_newton_ik(x_path)
        time_n = time.time() - start
        print(f"{time_n*1000:.1f}ms")
        
        print("Running Damped Newton IK (SOTA)... ", end='', flush=True)
        start = time.time()
        res_damped = damped_newton_ik(x_path)
        time_d = time.time() - start
        print(f"{time_d*1000:.1f}ms")
        
        print("Running φ-Surfer IK... ", end='', flush=True)
        start = time.time()
        res_phi = phi_surfer_ik(x_path)
        time_p = time.time() - start
        print(f"{time_p*1000:.1f}ms\n")
        
        # Results
        print("RESULTS:")
        print("-" * 90)
        print(f"{'Metric':<25} {'Naive Newton':<20} {'Damped Newton':<20} {'φ-Surfer':<20}")
        print("-" * 90)
        print(f"{'Stalls (Freeze)':<25} {res_newton['stalls']:<20} {res_damped['stalls']:<20} {res_phi['stalls']:<20}")
        print(f"{'Wild Jumps':<25} {res_newton['wild_jumps']:<20} {res_damped['wild_jumps']:<20} {res_phi['wild_jumps']:<20}")
        print(f"{'Failures':<25} {res_newton['failures']:<20} {res_damped['failures']:<20} {res_phi['failures']:<20}")
        print(f"{'Success Rate (%)':<25} {res_newton['success_rate']*100:<20.1f} {res_damped['success_rate']*100:<20.1f} {res_phi['success_rate']*100:<20.1f}")
        print(f"{'Runtime (ms)':<25} {time_n*1000:<20.1f} {time_d*1000:<20.1f} {time_p*1000:<20.1f}")
        print("=" * 90)
        
        # Winner
        if res_phi['stalls'] < res_damped['stalls'] * 0.7:
            print(f"\n🏆 WINNER: φ-Surfer")
            print(f"   ✨ Maintained velocity through singularity!")
            print(f"   ({res_phi['stalls']} stalls vs {res_damped['stalls']} for Damped Newton)")
        
        # Visualize
        results = {
            'x_path': x_path,
            'newton': res_newton,
            'damped': res_damped,
            'phi': res_phi,
            'desc': desc
        }
        
        save_path = f'/home/midori/gits/3d-newton-fractal/robot_{test_name}.png'
        visualize_robot_results(results, save_path)
    
    print(f"\n{'='*80}")
    print("ROBOT GAUNTLET COMPLETE!")
    print(f"{'='*80}\n")
