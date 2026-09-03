import numpy as np
import time


# ============================================================
# DOUBLE PENDULUM CHAOS DATASET GENERATOR
#
# Features:
#   - Correct double-pendulum equations of motion
#   - Vectorized RK4 integration
#   - Largest finite-time Lyapunov exponent (FTLE)
#   - Energy conservation diagnostics
#   - Reproducible random initial conditions
#   - float64 physics / float32 dataset storage
#
# State:
#   [theta1, theta2, omega1, omega2]
#
# Angles are measured from the DOWNWARD vertical.
# Angles are stored in radians.
#
# The Lyapunov calculation uses two nearby trajectories:
#   reference trajectory
#   shadow trajectory
#
# The shadow trajectory is repeatedly renormalized so that
# numerical separation does not become too large.
# ============================================================


# ============================================================
# Equations of Motion
# ============================================================
initial_time_entire_code =time.time()
def derivatives_batch(states, m1, m2, L1, L2, g):
    """
    Calculate derivatives for many double pendulums
    simultaneously.

    states shape:
        (N, 4)

    Columns:
        0 = theta1
        1 = theta2
        2 = omega1
        3 = omega2

    Returns:
        shape (N, 4)
        [omega1, omega2, alpha1, alpha2]
    """

    theta1 = states[:, 0]
    theta2 = states[:, 1]

    omega1 = states[:, 2]
    omega2 = states[:, 3]

    delta = theta2 - theta1

    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)

    # --------------------------------------------------------
    # Mass matrix:
    #
    # A * alpha1 + B * alpha2 = rhs1
    # B * alpha1 + C * alpha2 = rhs2
    # --------------------------------------------------------

    A = (m1 + m2) * L1**2

    B = m2 * L1 * L2 * cos_delta

    C = m2 * L2**2

    # --------------------------------------------------------
    # Correct equations of motion
    # --------------------------------------------------------

    rhs1 = (
        m2 * L1 * L2 * omega2**2 * sin_delta
        - (m1 + m2) * g * L1 * np.sin(theta1)
    )

    rhs2 = (
        -m2 * L1 * L2 * omega1**2 * sin_delta
        - m2 * g * L2 * np.sin(theta2)
    )

    # --------------------------------------------------------
    # Explicit solution of 2x2 system
    # --------------------------------------------------------

    determinant = A * C - B**2

    alpha1 = (
        rhs1 * C - B * rhs2
    ) / determinant

    alpha2 = (
        A * rhs2 - B * rhs1
    ) / determinant

    return np.column_stack((
        omega1,
        omega2,
        alpha1,
        alpha2
    ))


# ============================================================
# RK4
# ============================================================

def rk4_step_batch(
    states,
    dt,
    m1,
    m2,
    L1,
    L2,
    g
):
    """
    One RK4 step for all trajectories simultaneously.
    """

    k1 = derivatives_batch(
        states,
        m1, m2, L1, L2, g
    )

    k2 = derivatives_batch(
        states + 0.5 * dt * k1,
        m1, m2, L1, L2, g
    )

    k3 = derivatives_batch(
        states + 0.5 * dt * k2,
        m1, m2, L1, L2, g
    )

    k4 = derivatives_batch(
        states + dt * k3,
        m1, m2, L1, L2, g
    )

    return states + (
        dt / 6.0
    ) * (
        k1
        + 2.0 * k2
        + 2.0 * k3
        + k4
    )


# ============================================================
# Energy
# ============================================================

def compute_energy_batch(
    states,
    m1,
    m2,
    L1,
    L2,
    g
):
    """
    Total mechanical energy of each trajectory.
    """

    theta1 = states[:, 0]
    theta2 = states[:, 1]

    omega1 = states[:, 2]
    omega2 = states[:, 3]

    # --------------------------------------------------------
    # Kinetic energy
    # --------------------------------------------------------

    KE1 = (
        0.5
        * m1
        * L1**2
        * omega1**2
    )

    KE2 = 0.5 * m2 * (
        L1**2 * omega1**2
        + L2**2 * omega2**2
        + 2.0
        * L1
        * L2
        * omega1
        * omega2
        * np.cos(theta1 - theta2)
    )

    # --------------------------------------------------------
    # Potential energy
    # --------------------------------------------------------

    y1 = -L1 * np.cos(theta1)

    y2 = (
        y1
        - L2 * np.cos(theta2)
    )

    PE1 = m1 * g * y1
    PE2 = m2 * g * y2

    return KE1 + KE2 + PE1 + PE2


# ============================================================
# Space-Filling Initial States
# ============================================================

def random_initial_states(
    num_trajectories,
    rng
):
    """
    Generate space-filling initial conditions with a lightweight
    Latin-hypercube-style sampler.

    theta1, theta2:
        [-pi, pi]

    omega1, omega2:
        [-1, 1] rad/s
    """

    if num_trajectories <= 0:
        raise ValueError(
            "num_trajectories must be greater than 0."
        )

    states = np.empty(
        (num_trajectories, 4),
        dtype=np.float64
    )

    bounds = np.array(
        [
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-1.0, 1.0],
            [-1.0, 1.0]
        ],
        dtype=np.float64
    )

    for dimension, (lower, upper) in enumerate(bounds):
        bins = (
            np.arange(num_trajectories, dtype=np.float64)
            + 0.5
        ) / num_trajectories

        rng.shuffle(bins)

        states[:, dimension] = (
            lower
            + bins * (upper - lower)
        )

    return states


# ============================================================
# Lyapunov Initial Perturbation
# ============================================================

def create_shadow_states(
    states,
    delta0,
    characteristic_time,
    rng
):
    """
    Create nearby shadow trajectories.

    A physically useful scaled norm is used because:
        theta -> radians
        omega -> radians/second

    Multiplying angular velocity differences by a
    characteristic time gives comparable units.

    The characteristic time is approximately:
        sqrt(L1 / g)
    """

    N = states.shape[0]

    # Random direction in 4D tangent space
    direction = rng.normal(
        size=(N, 4)
    )

    # Scale angular velocity components
    direction[:, 2:] *= characteristic_time

    norms = np.linalg.norm(
        direction,
        axis=1
    )

    direction /= norms[:, None]

    # Convert scaled perturbation back to state units
    perturbation = direction.copy()

    perturbation[:, 2:] /= characteristic_time

    perturbation *= delta0

    return states + perturbation


# ============================================================
# Calculate Lyapunov Exponents
# ============================================================

def update_lyapunov(
    reference,
    shadow,
    sums,
    delta0,
    characteristic_time
):
    """
    Measure separation between reference and shadow
    trajectories, accumulate log growth, and renormalize.

    Returns:
        updated shadow states
        updated sums
    """

    difference = shadow - reference

    # Scale angular velocities so the norm is dimensionally
    # consistent.
    scaled_difference = difference.copy()

    scaled_difference[:, 2:] *= characteristic_time

    distances = np.linalg.norm(
        scaled_difference,
        axis=1
    )

    # Avoid log(0)
    distances = np.maximum(
        distances,
        1e-300
    )

    # Accumulate logarithmic growth
    sums += np.log(
        distances / delta0
    )

    # --------------------------------------------------------
    # Renormalize
    # --------------------------------------------------------

    scaled_difference *= (
        delta0 / distances
    )[:, None]

    # Convert scaled angular velocity differences back
    # to physical state units.
    scaled_difference[:, 2:] /= (
        characteristic_time
    )

    shadow = reference + scaled_difference

    return shadow, sums


# ============================================================
# Input Validation
# ============================================================

def validate_parameters(
    m1,
    m2,
    L1,
    L2,
    T,
    dt1,
    dt,
    num_trajectories,
    lyapunov_interval,
    delta0
):

    if m1 <= 0:
        raise ValueError(
            "m1 must be greater than 0."
        )

    if m2 <= 0:
        raise ValueError(
            "m2 must be greater than 0."
        )

    if L1 <= 0:
        raise ValueError(
            "L1 must be greater than 0."
        )

    if L2 <= 0:
        raise ValueError(
            "L2 must be greater than 0."
        )

    if T <= 0:
        raise ValueError(
            "T must be greater than 0."
        )

    if dt1 <= 0:
        raise ValueError(
            "dt1 must be greater than 0."
        )

    if dt <= 0:
        raise ValueError(
            "dt must be greater than 0."
        )

    if dt1 > dt:
        raise ValueError(
            "dt1 must be less than or equal to dt."
        )

    steps_per_saved_state = int(round(dt / dt1))

    if steps_per_saved_state < 1:
        raise ValueError(
            "dt/dt1 must be at least 1."
        )

    if not np.isclose(
        steps_per_saved_state * dt1,
        dt,
        rtol=0.0,
        atol=1e-12
    ):
        raise ValueError(
            "dt must be an integer multiple of dt1."
        )

    if dt >= T:
        raise ValueError(
            "dt must be smaller than T."
        )

    if num_trajectories <= 0:
        raise ValueError(
            "Number of trajectories must be > 0."
        )

    if lyapunov_interval <= 0:
        raise ValueError(
            "Lyapunov interval must be > 0."
        )

    if delta0 <= 0:
        raise ValueError(
            "Lyapunov perturbation must be > 0."
        )


# ============================================================
# Main Dataset Generator
# ============================================================

def generate_dataset(
    m1,
    m2,
    L1,
    L2,
    T,
    dt1=0.00001,
    dt=0.001,
    num_trajectories=2000,#use this to make more accuracy and more time
    g=9.80665,
    seed=42,
    lyapunov_interval=10,
    delta0=1e-8,
    out_path="double_pendulum_chaos_dataset.npz"
):
    """
    Generate a chaos-focused double-pendulum dataset.

    Parameters
    ----------
    m1, m2:
        masses in kg

    L1, L2:
        lengths in meters

    T:
        simulation duration in seconds

    dt1:
        high-resolution RK4 physics timestep

    dt:
        timestep between saved ML states

    num_trajectories:
        number of random trajectories

    g:
        gravitational acceleration

    seed:
        random seed

    lyapunov_interval:
        number of saved ML steps between Lyapunov
        renormalizations.

        Default = 10

    delta0:
        initial dimensionless scaled perturbation.

    out_path:
        output .npz file
    """

    validate_parameters(
        m1,
        m2,
        L1,
        L2,
        T,
        dt1,
        dt,
        num_trajectories,
        lyapunov_interval,
        delta0
    )

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Number of steps
    # --------------------------------------------------------

    steps = int(
        round(T / dt)
    )

    actual_T = steps * dt

    physics_steps_per_ml_step = int(
        round(dt / dt1)
    )
    total_physics_steps = (
        steps * physics_steps_per_ml_step
    )

    if lyapunov_interval > steps:
        raise ValueError(
            "Lyapunov interval cannot exceed the number of saved steps."
        )

    # --------------------------------------------------------
    # Random generator
    # --------------------------------------------------------

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Characteristic timescale
    # --------------------------------------------------------

    characteristic_time = np.sqrt(
        L1 / g
    )

    # --------------------------------------------------------
    # Generate initial conditions
    # --------------------------------------------------------

    states = random_initial_states(
        num_trajectories,
        rng
    )

    initial_states = states.copy()

    # --------------------------------------------------------
    # Create nearby shadow trajectories
    # --------------------------------------------------------

    shadow_rng = np.random.default_rng(
        None if seed is None else seed + 1
    )

    shadow_states = create_shadow_states(
        states,
        delta0,
        characteristic_time,
        shadow_rng
    )

    # --------------------------------------------------------
    # Allocate dataset
    # --------------------------------------------------------

    trajectories = np.empty(
        (
            num_trajectories,
            steps + 1,
            4
        ),
        dtype=np.float32
    )

    trajectories[:, 0, :] = states

    # --------------------------------------------------------
    # Initial energies
    # --------------------------------------------------------

    initial_energies = compute_energy_batch(
        states,
        m1,
        m2,
        L1,
        L2,
        g
    )

    # --------------------------------------------------------
    # Lyapunov accumulators
    # --------------------------------------------------------

    lyapunov_sums = np.zeros(
        num_trajectories,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("DOUBLE PENDULUM CHAOS DATASET")
    print("=" * 65)

    print(f"m1                         : {m1} kg")
    print(f"m2                         : {m2} kg")
    print(f"L1                         : {L1} m")
    print(f"L2                         : {L2} m")
    print(f"g                          : {g} m/s^2")
    print(f"T                          : {actual_T} s")
    print(f"Physics dt1                : {dt1} s")
    print(f"ML / saved dt              : {dt} s")
    print(f"Physics substeps / ML step: {physics_steps_per_ml_step:,}")
    print(f"Saved ML steps / trajectory: {steps:,}")
    print(f"Trajectories               : {num_trajectories:,}")
    print(f"Total physics RK4 steps   : {total_physics_steps * num_trajectories:,}")

    print()
    print("LYAPUNOV SETTINGS")
    print("-----------------")
    print(
        f"Renormalization interval   : "
        f"{lyapunov_interval} steps"
    )

    print(
        f"Renormalization time       : "
        f"{lyapunov_interval * dt} s"
    )

    print(
        f"Initial perturbation       : "
        f"{delta0:.1e}"
    )

    print(
        f"Characteristic time        : "
        f"{characteristic_time:.6f} s"
    )

    print()
    print(f"Random seed                : {seed}")
    print("=" * 65)
    print()

    # --------------------------------------------------------
    # Main integration
    # --------------------------------------------------------

    progress_interval = max(
        1,
        steps // 20
    )

    for step in range(steps):

        # Advance both trajectories with high-resolution dt1.
        # Only the final state at the ML timestep is stored.
        # For dt1=0.00001 and dt=0.001, this is 100 RK4
        # physics steps for every one saved ML state.
        for _ in range(physics_steps_per_ml_step):
            states = rk4_step_batch(
                states, dt1, m1, m2, L1, L2, g
            )
            shadow_states = rk4_step_batch(
                shadow_states, dt1, m1, m2, L1, L2, g
            )

        trajectories[:, step + 1, :] = states

        # ----------------------------------------------------
        # Lyapunov renormalization
        # ----------------------------------------------------

        if (
            (step + 1) % lyapunov_interval == 0
        ):

            shadow_states, lyapunov_sums = (
                update_lyapunov(
                    states,
                    shadow_states,
                    lyapunov_sums,
                    delta0,
                    characteristic_time
                )
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (step + 1) % progress_interval == 0
            or step == steps - 1
        ):

            percent = (
                100.0
                * (step + 1)
                / steps
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            estimated_total = (
                elapsed
                * steps
                / (step + 1)
            )

            remaining = max(
                0.0,
                estimated_total - elapsed
            )

            print(
                f"\rProgress: {percent:6.2f}% | "
                f"Elapsed: {elapsed / 60:6.1f} min | "
                f"ETA: {remaining / 60:6.1f} min",
                end="",
                flush=True
            )

    # Include any final partial Lyapunov interval so that the
    # accumulated growth covers the full simulated time.
    if steps % lyapunov_interval != 0:
        shadow_states, lyapunov_sums = update_lyapunov(
            states,
            shadow_states,
            lyapunov_sums,
            delta0,
            characteristic_time
        )

    print()
    print()

    # --------------------------------------------------------
    # Final energy
    # --------------------------------------------------------

    final_energies = compute_energy_batch(
        states,
        m1,
        m2,
        L1,
        L2,
        g
    )

    # --------------------------------------------------------
    # Energy errors
    # --------------------------------------------------------

    absolute_energy_errors = (
        final_energies
        - initial_energies
    )

    relative_energy_errors = (
        np.abs(absolute_energy_errors)
        / np.maximum(
            np.abs(initial_energies),
            1e-12
        )
    )

    # --------------------------------------------------------
    # Largest finite-time Lyapunov exponent
    #
    # Sum(log growth) / total elapsed time
    # --------------------------------------------------------

    lyapunov_exponents = (
        lyapunov_sums
        / actual_T
    )

    # --------------------------------------------------------
    # Classification
    #
    # We save the raw exponent as the scientifically useful
    # quantity. Classification is based on its sign.
    #
    # A tiny numerical tolerance avoids calling values that
    # are effectively zero "chaotic".
    # --------------------------------------------------------

    chaos_tolerance = 1e-3

    chaotic = (
        lyapunov_exponents
        > chaos_tolerance
    )

    nonchaotic = (
        lyapunov_exponents
        < -chaos_tolerance
    )

    borderline = ~(
        chaotic | nonchaotic
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    mean_abs_energy_error = np.mean(
        np.abs(absolute_energy_errors)
    )

    max_abs_energy_error = np.max(
        np.abs(absolute_energy_errors)
    )

    mean_relative_energy_error = np.mean(
        relative_energy_errors
    )

    max_relative_energy_error = np.max(
        relative_energy_errors
    )

    num_chaotic = np.sum(
        chaotic
    )

    num_nonchaotic = np.sum(
        nonchaotic
    )

    num_borderline = np.sum(
        borderline
    )

    total_time = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.savez(
        out_path,

        # -----------------------------------------------
        # Main trajectory data
        # -----------------------------------------------

        trajectories=trajectories,

        # -----------------------------------------------
        # Initial states
        # -----------------------------------------------

        initial_states=(
            initial_states.astype(
                np.float32
            )
        ),

        # -----------------------------------------------
        # Energy diagnostics
        # -----------------------------------------------

        energy_errors=(
            absolute_energy_errors
        ),

        relative_energy_errors=(
            relative_energy_errors
        ),

        # Backwards-compatible name
        errors=absolute_energy_errors,

        # -----------------------------------------------
        # Chaos diagnostics
        # -----------------------------------------------

        lyapunov_exponents=(
            lyapunov_exponents
        ),

        chaotic=chaotic,

        nonchaotic=nonchaotic,

        borderline=borderline,

        chaos_tolerance=chaos_tolerance,

        # -----------------------------------------------
        # Physical parameters
        # -----------------------------------------------

        m1=m1,
        m2=m2,
        L1=L1,
        L2=L2,
        g=g,

        # -----------------------------------------------
        # Simulation parameters
        # -----------------------------------------------

        T=actual_T,
        dt=dt,
        dt1=dt1,
        num_trajectories=(
            num_trajectories
        ),

        # -----------------------------------------------
        # Lyapunov parameters
        # -----------------------------------------------

        lyapunov_interval=(
            lyapunov_interval
        ),

        lyapunov_delta0=delta0,

        characteristic_time=(
            characteristic_time
        ),

        # -----------------------------------------------
        # Reproducibility
        # -----------------------------------------------

        seed=(
            -1
            if seed is None
            else seed
        )
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("=" * 65)
    print("DATASET COMPLETE")
    print("=" * 65)

    print(
        f"Output file: {out_path}"
    )

    print(
        f"Dataset shape: "
        f"{trajectories.shape}"
    )

    print()
    print("ENERGY CONSERVATION")
    print("-------------------")

    print(
        f"Mean |ΔE|: "
        f"{mean_abs_energy_error:.6e} J"
    )

    print(
        f"Max |ΔE|:  "
        f"{max_abs_energy_error:.6e} J"
    )

    print(
        f"Mean relative energy error: "
        f"{mean_relative_energy_error:.6e}"
    )

    print(
        f"Max relative energy error:  "
        f"{max_relative_energy_error:.6e}"
    )

    print()
    print("LYAPUNOV ANALYSIS")
    print("-----------------")

    print(
        f"Mean largest FTLE: "
        f"{np.mean(lyapunov_exponents):.6e} 1/s"
    )

    print(
        f"Minimum largest FTLE: "
        f"{np.min(lyapunov_exponents):.6e} 1/s"
    )

    print(
        f"Maximum largest FTLE: "
        f"{np.max(lyapunov_exponents):.6e} 1/s"
    )

    print()
    print(
        f"Positive LLE (> {chaos_tolerance}): "
        f"{num_chaotic:,}"
    )

    print(
        f"Negative LLE (< -{chaos_tolerance}): "
        f"{num_nonchaotic:,}"
    )

    print(
        f"Near zero: "
        f"{num_borderline:,}"
    )

    print(
        f"Estimated chaotic fraction: "
        f"{num_chaotic / num_trajectories:.2%}"
    )

    print()
    print(
        f"Total runtime: "
        f"{total_time / 60:.2f} minutes"
    )

    print("=" * 65)
    print()


# ============================================================
# Load Dataset
# ============================================================

def load_dataset(path):
    """
    Load the chaos dataset.
    """

    data = np.load(path)

    print()
    print("DATASET INFORMATION")
    print("--------------------")

    print(
        "Available arrays:"
    )

    for key in data.files:
        value = data[key]

        if hasattr(value, "shape"):
            print(
                f"  {key:25s} "
                f"shape={value.shape}"
            )
        else:
            print(
                f"  {key:25s}"
            )

    return data


# ============================================================
# User Input
# ============================================================

def get_user_input():

    print()
    print("=" * 65)
    print("DOUBLE PENDULUM CHAOS DATASET SETUP")
    print("=" * 65)
    print()

    # --------------------------------------------------------
    # Physical parameters
    # --------------------------------------------------------

    m1 = float(
        input(
            "Mass of first bob (kg): "
        )
    )

    m2 = float(
        input(
            "Mass of second bob (kg): "
        )
    )

    L1 = float(
        input(
            "Length of first pendulum (m): "
        )
    )

    L2 = float(
        input(
            "Length of second pendulum (m): "
        )
    )

    # --------------------------------------------------------
    # Simulation time
    # --------------------------------------------------------

    T = float(
        input(
            "Simulation time T (seconds): "
        )
    )

    # --------------------------------------------------------
    # High-resolution physics timestep
    # --------------------------------------------------------

    dt1_input = input(
        "Physics timestep dt1 [default 0.00001]: "
    ).strip()

    if dt1_input == "":
        dt1 = 0.00001
    else:
        dt1 = float(dt1_input)

    # ML / saved timestep is fixed at 0.001 s.
    dt = 0.001
    print(f"ML timestep dt is fixed at {dt} s.")

    # --------------------------------------------------------
    # Number of trajectories
    # --------------------------------------------------------

    n_input = input(
        "Number of trajectories [default 1000]: "
    ).strip()

    if n_input == "":
        num_trajectories = 1000
    else:
        num_trajectories = int(n_input)

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    seed_input = input(
        "Random seed [default 42]: "
    ).strip()

    if seed_input == "":
        seed = 42
    else:
        seed = int(
            seed_input
        )

    # --------------------------------------------------------
    # Lyapunov interval
    # --------------------------------------------------------

    interval_input = input(
        "Lyapunov renormalization interval "
        "in steps [default 10]: "
    ).strip()

    if interval_input == "":
        lyapunov_interval = 10
    else:
        lyapunov_interval = int(
            interval_input
        )

    # --------------------------------------------------------
    # Output filename
    # --------------------------------------------------------

    filename_input = input(
        "Output filename "
        "[default double_pendulum_chaos_dataset.npz]: "
    ).strip()

    if filename_input == "":
        out_path = (
            "double_pendulum_chaos_dataset.npz"
        )
    else:
        out_path = filename_input

    return (
        m1,
        m2,
        L1,
        L2,
        T,
        dt1,
        dt,
        num_trajectories,
        seed,
        lyapunov_interval,
        out_path
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    try:

        (
            m1,
            m2,
            L1,
            L2,
            T,
            dt1,
            dt,
            num_trajectories,
            seed,
            lyapunov_interval,
            out_path
        ) = get_user_input()

        generate_dataset(
            m1=m1,
            m2=m2,
            L1=L1,
            L2=L2,
            T=T,
            dt1=dt1,
            dt=dt,
            num_trajectories=num_trajectories,
            g=9.80665,
            seed=seed,
            lyapunov_interval=lyapunov_interval,
            delta0=1e-8,
            out_path=out_path
        )

    except KeyboardInterrupt:

        print()
        print()
        print("Simulation cancelled.")
        raise SystemExit(1)

    except ValueError as error:

        print()
        print(
            f"Input error: {error}"
        )
        raise SystemExit(1)

    except Exception as error:

        print()
        print()
        print(
            f"Unexpected error: {error}"
        )
        raise SystemExit(1)







































































































































































































# ============================================================
# NEURAL NETWORK
# PHYSICS-INFORMED LSTM
#
# Physics integration uses dt1.
# ML data spacing is dt = 0.001 s.
#
# FEATURES:
# - LSTM sequence prediction
# - Delta-state prediction
# - Cartesian x/y loss
# - Double-pendulum physics loss
# - Kinematic consistency loss
# - Energy conservation loss
# - FTLE-weighted training
# - CUDA / AMP support
# - AdamW
# - learning-rate schedule
# - Gradient clipping
# - Early stopping
# ============================================================

import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
# ============================================================
# CONFIGURATION SETTINGS- UPDATE TO CHANGE LENGTH/ACCURACY
# ============================================================
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################

DATASET_PATH = globals().get(
    "out_path",
    "double_pendulum_chaos_dataset.npz"
)

MODEL_PATH = "double_pendulum_pilstm_best.pt"


# ------------------------------------------------------------
# Sequence
# ------------------------------------------------------------

SEQUENCE_LENGTH = 50




# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

HIDDEN_SIZE = 320
NUM_LAYERS = 3
DROPOUT = 0.08


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
##############################################################
##############################################################
##############################################################
##############################################################EPOCHS
EPOCHS = 100
##############################################################
##############################################################
##############################################################
##############################################################
BATCH_SIZE = 512

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 1e-6

SAMPLES_PER_EPOCH = 300_000

VALIDATION_SAMPLES = 30_000

PATIENCE = 30

# Multi-step rollout training is intentionally disabled to keep the
# objective stable, fast, and consistent with the single-step physics
# model used for training and validation.

# ============================================================
# LOSS WEIGHTS
# ============================================================


# Direct state prediction.
STATE_LOSS_WEIGHT = 0.2585344729

# Cartesian x/y position accuracy.
POSITION_LOSS_WEIGHT = 1.466155007

# Consistency with differentiable RK4 physics.
PHYSICS_LOSS_WEIGHT = 0.05609765143

# Mechanical-energy conservation.
ENERGY_LOSS_WEIGHT = 0.01194651913


# ============================================================
# FTLE WEIGHTING
# ============================================================

# High-FTLE trajectories are harder and therefore receive
# greater training weight.
#
# Weight is approximately:
#
#     normalized_FTLE_weight = FTLE / mean_positive_FTLE
#
# followed by clipping.
#
# This prevents one extremely chaotic trajectory from
# completely dominating training.

FTLE_WEIGHT_MIN = 0.868535473
FTLE_WEIGHT_MAX = 1.99699117

FTLE_EPSILON = 1e-8


# ============================================================
# ACCURACY THRESHOLD
# ============================================================

POSITION_TOLERANCE = 1e-5


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print("=" * 75)
print("Double pendulum model setup")
print("=" * 75)

print(
    f"The active device is {DEVICE}."
)

if DEVICE.type == "cuda":
    print(
        "The graphics processing unit is "
        f"{torch.cuda.get_device_name(0)}."
    )

print("=" * 75)
print()


# ============================================================
# LOAD DATASET
# ============================================================

print("The dataset is loading now.")

data = np.load(
    DATASET_PATH,
    allow_pickle=False
)


# ------------------------------------------------------------
# Main trajectory data
# ------------------------------------------------------------

trajectories = data[
    "trajectories"
]


# ------------------------------------------------------------
# Physical parameters
# ------------------------------------------------------------

m1 = float(
    data["m1"]
)

m2 = float(
    data["m2"]
)

L1 = float(
    data["L1"]
)

L2 = float(
    data["L2"]
)

g = float(
    data["g"]
)


# ------------------------------------------------------------
# ML timestep
# ------------------------------------------------------------

dt = float(
    data["dt"]
)

if not np.isfinite(dt) or dt <= 0.0:
    raise ValueError(
        "Dataset contains an invalid dt."
    )


if not np.isclose(
    dt,
    0.001,
    rtol=0.0,
    atol=1e-12
):
    raise ValueError(
        f"This ML code requires dt=0.001 s, "
        f"but the dataset contains dt={dt}."
    )


# ------------------------------------------------------------
# FTLE data
# ------------------------------------------------------------

if "lyapunov_exponents" not in data:
    raise KeyError(
        "Dataset does not contain "
        "'lyapunov_exponents'. "
        "Regenerate the dataset with FTLE diagnostics."
    )


lyapunov_exponents = np.asarray(
    data["lyapunov_exponents"],
    dtype=np.float64
)


# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------

num_trajectories = int(
    data["num_trajectories"]
)

T = float(
    data["T"]
)

if len(lyapunov_exponents) != num_trajectories:
    raise ValueError(
        "Number of FTLE values does not match "
        "number of trajectories."
    )


if not np.all(
    np.isfinite(lyapunov_exponents)
):
    raise ValueError(
        "FTLE array contains NaN or infinity."
    )


print(
    f"The loaded dataset has shape {trajectories.shape}."
)

print(
    f"There are {num_trajectories:,} trajectories in the dataset."
)

print(
    f"Each trajectory contains {trajectories.shape[1]:,} time steps."
)

print(
    f"The saved time step is {dt} seconds."
)

print(
    f"The first pendulum length is {L1} meters."
)

print(
    f"The second pendulum length is {L2} meters."
)

print(
    f"The mean largest finite-time Lyapunov exponent is "
    f"{np.mean(lyapunov_exponents):.6e} per second."
)

print(
    f"The minimum largest finite-time Lyapunov exponent is "
    f"{np.min(lyapunov_exponents):.6e} per second."
)

print(
    f"Maximum FTLE: "
    f"{np.max(lyapunov_exponents):.6e} 1/s"
)

print()


# ============================================================
# PHYSICAL FUNCTIONS
# ============================================================

def state_to_xy(
    states
):
    """
    Convert:

        [theta1, theta2, omega1, omega2]

    into:

        [x1, y1, x2, y2]
    """

    theta1 = states[..., 0]

    theta2 = states[..., 1]


    x1 = (
        L1
        * np.sin(theta1)
    )

    y1 = (
        -L1
        * np.cos(theta1)
    )


    x2 = (
        x1
        + L2
        * np.sin(theta2)
    )

    y2 = (
        y1
        - L2
        * np.cos(theta2)
    )


    return np.stack(
        [
            x1,
            y1,
            x2,
            y2
        ],
        axis=-1
    )


# ============================================================
# STATE -> NETWORK FEATURES
# ============================================================

def state_to_features(
    states
):
    """
    Six neural-network input features:

        sin(theta1)
        cos(theta1)
        sin(theta2)
        cos(theta2)
        omega1
        omega2
    """

    theta1 = states[..., 0]

    theta2 = states[..., 1]

    omega1 = states[..., 2]

    omega2 = states[..., 3]


    return np.stack(
        [
            np.sin(theta1),
            np.cos(theta1),
            np.sin(theta2),
            np.cos(theta2),
            omega1,
            omega2
        ],
        axis=-1
    )


def physical_state_to_features(
    states
):
    """
    Convert a physical state tensor:

        [theta1, theta2, omega1, omega2]

    into normalized neural-network features:

        [sin(theta1), cos(theta1), sin(theta2), cos(theta2), omega1, omega2]
    """

    theta1 = states[..., 0]
    theta2 = states[..., 1]
    omega1 = states[..., 2]
    omega2 = states[..., 3]

    return torch.stack(
        [
            torch.sin(theta1),
            torch.cos(theta1),
            torch.sin(theta2),
            torch.cos(theta2),
            omega1,
            omega2
        ],
        dim=-1
    )


# ============================================================
# WRAPPED ANGLE DIFFERENCE
# ============================================================

def wrapped_angle_difference(
    a,
    b
):
    """
    Difference a-b wrapped into [-pi, pi].
    """

    return (
        (a - b + np.pi)
        % (2.0 * np.pi)
        - np.pi
    )


def wrapped_angle_difference_torch(
    a,
    b
):
    """Return the shortest differentiable circular difference a-b."""

    return torch.atan2(
        torch.sin(a - b),
        torch.cos(a - b)
    )


# ============================================================
# TARGET DELTA
# ============================================================

def state_delta(
    current,
    target
):
    """
    Calculate:

        target - current

    with angular differences wrapped.
    """

    delta = (
        target
        - current
    )

    delta = delta.copy()


    delta[..., 0] = (
        wrapped_angle_difference(
            target[..., 0],
            current[..., 0]
        )
    )

    delta[..., 1] = (
        wrapped_angle_difference(
            target[..., 1],
            current[..., 1]
        )
    )


    return delta


# ============================================================
# APPLY DELTA
# ============================================================

def apply_delta(
    current,
    delta
):
    """
    Apply a physical state delta to a physical state.
    """

    result = (
        current
        + delta
    )


    result[..., 0] = (
        result[..., 0]
        + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


    result[..., 1] = (
        result[..., 1]
        + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


    return result


# ============================================================
# DOUBLE-PENDULUM ACCELERATIONS
# ============================================================

def double_pendulum_accelerations_torch(
    theta1,
    theta2,
    omega1,
    omega2
):
    """
    Differentiable double-pendulum equations.

    Returns:

        alpha1
        alpha2
    """

    delta_theta = (
        theta2
        - theta1
    )

    sin_delta = torch.sin(
        delta_theta
    )

    cos_delta = torch.cos(
        delta_theta
    )


    # --------------------------------------------------------
    # Mass-matrix coefficients
    # --------------------------------------------------------

    A = (
        (m1 + m2)
        * L1
        * L1
    )

    B = (
        m2
        * L1
        * L2
        * cos_delta
    )

    C = (
        m2
        * L2
        * L2
    )


    # --------------------------------------------------------
    # Right-hand sides
    # --------------------------------------------------------

    rhs1 = (
        m2
        * L1
        * L2
        * omega2
        * omega2
        * sin_delta
        -
        (m1 + m2)
        * g
        * L1
        * torch.sin(theta1)
    )


    rhs2 = (
        -m2
        * L1
        * L2
        * omega1
        * omega1
        * sin_delta
        -
        m2
        * g
        * L2
        * torch.sin(theta2)
    )


    # --------------------------------------------------------
    # Determinant
    # --------------------------------------------------------

    determinant = (
        A * C
        - B * B
    )


    # --------------------------------------------------------
    # Solve 2x2 system
    # --------------------------------------------------------

    alpha1 = (
        rhs1 * C
        - B * rhs2
    ) / determinant


    alpha2 = (
        A * rhs2
        - B * rhs1
    ) / determinant


    return (
        alpha1,
        alpha2
    )


# ============================================================
# MECHANICAL ENERGY
# ============================================================

def torch_energy(
    theta1,
    theta2,
    omega1,
    omega2
):
    """
    Total mechanical energy.

    Same physical convention as the dataset generator.
    """

    # --------------------------------------------------------
    # Kinetic energy of mass 1
    # --------------------------------------------------------

    KE1 = (
        0.5
        * m1
        * L1**2
        * omega1**2
    )


    # --------------------------------------------------------
    # Kinetic energy of mass 2
    # --------------------------------------------------------

    KE2 = (
        0.5
        * m2
        * (
            L1**2
            * omega1**2
            +
            L2**2
            * omega2**2
            +
            2.0
            * L1
            * L2
            * omega1
            * omega2
            * torch.cos(
                theta1 - theta2
            )
        )
    )


    # --------------------------------------------------------
    # Heights
    # --------------------------------------------------------

    y1 = (
        -L1
        * torch.cos(theta1)
    )

    y2 = (
        y1
        - L2
        * torch.cos(theta2)
    )


    # --------------------------------------------------------
    # Potential energy
    # --------------------------------------------------------

    PE1 = (
        m1
        * g
        * y1
    )

    PE2 = (
        m2
        * g
        * y2
    )


    return (
        KE1
        + KE2
        + PE1
        + PE2
    )


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

indices = np.arange(
    num_trajectories
)


rng = np.random.default_rng(
    SEED
)

rng.shuffle(
    indices
)


train_end = int(
    0.80
    * num_trajectories
)

validation_end = int(
    0.90
    * num_trajectories
)


train_indices = indices[
    :train_end
]

validation_indices = indices[
    train_end:validation_end
]

test_indices = indices[
    validation_end:
]


if len(train_indices) == 0:
    raise ValueError(
        "Training split is empty."
    )

if len(validation_indices) == 0:
    raise ValueError(
        "Validation split is empty."
    )

if len(test_indices) == 0:
    raise ValueError(
        "Test split is empty."
    )


print("The data is being split into training, validation, and test sets.")
print(
    f"The training set contains {len(train_indices):,} trajectories."
)
print(
    f"The validation set contains {len(validation_indices):,} trajectories."
)
print(
    f"The test set contains {len(test_indices):,} trajectories."
)
print()


# ============================================================
# FTLE SAMPLE WEIGHTS
# ============================================================

def make_ftle_weights(
    exponents
):
    """
    Convert FTLE values into stable training weights.

    Positive / strongly chaotic trajectories receive more
    weight.

    Non-positive FTLE values are not given zero weight because
    the model still needs to learn the complete physical system.
    """

    exponents = np.asarray(
        exponents,
        dtype=np.float64
    )


    # --------------------------------------------------------
    # Positive FTLE magnitude
    # --------------------------------------------------------

    positive_part = np.maximum(
        exponents,
        0.0
    )


    positive_values = (
        positive_part[
            positive_part > FTLE_EPSILON
        ]
    )


    if positive_values.size > 0:

        reference = float(
            np.mean(
                positive_values
            )
        )

    else:

        reference = 1.0


    # --------------------------------------------------------
    # Base weight
    # --------------------------------------------------------

    weights = (
        1.0
        +
        positive_part
        / (
            reference
            + FTLE_EPSILON
        )
    )


    # --------------------------------------------------------
    # Normalize around approximately 1
    # --------------------------------------------------------

    weights = (
        weights
        / np.mean(weights)
    )


    # --------------------------------------------------------
    # Clip extreme values
    # --------------------------------------------------------

    weights = np.clip(
        weights,
        FTLE_WEIGHT_MIN,
        FTLE_WEIGHT_MAX
    )


    return weights.astype(
        np.float32
    )


ftle_weights = make_ftle_weights(
    lyapunov_exponents
)


print(
    "The training weights are being adjusted to give more influence to the most chaotic trajectories."
)

print(
    f"The smallest training weight is {np.min(ftle_weights):.4f}."
)

print(
    f"The largest training weight is {np.max(ftle_weights):.4f}."
)

print(
    f"The mean training weight is {np.mean(ftle_weights):.4f}."
)

print()


# ============================================================
# NORMALIZATION
# ============================================================

print(
    "The model inputs are being normalized so the training process remains stable and consistent."
)


train_data = trajectories[
    train_indices
].astype(
    np.float64
)


# ------------------------------------------------------------
# State deltas
# ------------------------------------------------------------

current_train = (
    train_data[:, :-1]
)

next_train = (
    train_data[:, 1:]
)


delta_train = state_delta(
    current_train,
    next_train
)


delta_mean = np.mean(
    delta_train,
    axis=(0, 1)
)


delta_std = np.std(
    delta_train,
    axis=(0, 1)
)


delta_std = np.maximum(
    delta_std,
    1e-10
)


if not (
    np.all(
        np.isfinite(delta_mean)
    )
    and
    np.all(
        np.isfinite(delta_std)
    )
):
    raise ValueError(
        "Delta normalization contains "
        "NaN or infinity."
    )


# ------------------------------------------------------------
# Input features
# ------------------------------------------------------------

feature_train = state_to_features(
    train_data
)


feature_mean = np.mean(
    feature_train,
    axis=(0, 1)
)


feature_std = np.std(
    feature_train,
    axis=(0, 1)
)


feature_std = np.maximum(
    feature_std,
    1e-8
)


if not (
    np.all(
        np.isfinite(feature_mean)
    )
    and
    np.all(
        np.isfinite(feature_std)
    )
):
    raise ValueError(
        "Feature normalization contains "
        "NaN or infinity."
    )


del train_data
del current_train
del next_train
del delta_train
del feature_train


print(
    "Normalization is complete."
)

print()


# ============================================================
# DATASET
# ============================================================

class PendulumDataset(
    Dataset
):

    def __init__(
        self,
        trajectories,
        trajectory_indices,
        sequence_length,
        samples_per_epoch,
        feature_mean,
        feature_std,
        delta_mean,
        delta_std,
        ftle_weights,
        seed
    ):

        self.trajectories = (
            trajectories
        )

        self.indices = (
            trajectory_indices
        )

        self.sequence_length = (
            sequence_length
        )

        self.samples_per_epoch = (
            samples_per_epoch
        )

        self.feature_mean = (
            feature_mean
        )

        self.feature_std = (
            feature_std
        )

        self.delta_mean = (
            delta_mean
        )

        self.delta_std = (
            delta_std
        )

        self.ftle_weights = (
            ftle_weights
        )


        if sequence_length < 1:
            raise ValueError(
                "sequence_length must be positive."
            )


        self.seed = int(
            seed
        )

        self.worker_rng = None

        self.max_start = (
            trajectories.shape[1]
            -
            sequence_length
            -
            1
        )


        if self.max_start < 0:
            raise ValueError(
                "Trajectory is too short for the requested sequence length."
            )


    def __len__(
        self
    ):

        return (
            self.samples_per_epoch
        )


    def __getitem__(
        self,
        index
    ):

        worker_info = (
            torch.utils.data
            .get_worker_info()
        )


        if worker_info is None:

            if self.worker_rng is None:

                self.worker_rng = (
                    np.random.default_rng(
                        self.seed
                    )
                )

        else:

            if self.worker_rng is None:

                self.worker_rng = (
                    np.random.default_rng(
                        self.seed
                        +
                        1_000_003
                        *
                        worker_info.id
                    )
                )


        # ----------------------------------------------------
        # Random trajectory
        # ----------------------------------------------------

        trajectory_index = (
            self.worker_rng.choice(
                self.indices
            )
        )


        # ----------------------------------------------------
        # Random starting point
        # ----------------------------------------------------

        start = (
            self.worker_rng.integers(
                0,
                self.max_start + 1
            )
        )


        end = (
            start
            +
            self.sequence_length
        )


        # ----------------------------------------------------
        # Sequence
        # ----------------------------------------------------

        sequence_states = (
            self.trajectories[
                trajectory_index,
                start:end
            ].astype(
                np.float64
            )
        )


        # ----------------------------------------------------
        # Current state
        # ----------------------------------------------------

        current_state = (
            sequence_states[-1]
        )


        # ----------------------------------------------------
        # Target state
        # ----------------------------------------------------

        target_state = (
            self.trajectories[
                trajectory_index,
                end
            ].astype(
                np.float64
            )
        )


        # ----------------------------------------------------
        # Input features
        # ----------------------------------------------------

        features = state_to_features(
            sequence_states
        )


        features = (
            features
            -
            self.feature_mean
        ) / self.feature_std


        # ----------------------------------------------------
        # Target delta
        # ----------------------------------------------------

        target_delta = state_delta(
            current_state,
            target_state
        )


        normalized_delta = (
            target_delta
            -
            self.delta_mean
        ) / self.delta_std


        # ----------------------------------------------------
        # Current features
        # ----------------------------------------------------

        current_features = (
            state_to_features(
                current_state
            )
        )


        current_features = (
            current_features
            -
            self.feature_mean
        ) / self.feature_std


        # ----------------------------------------------------
        # FTLE weight
        # ----------------------------------------------------

        sample_weight = (
            self.ftle_weights[
                trajectory_index
            ]
        )

        return (
            torch.tensor(
                features,
                dtype=torch.float32
            ),

            torch.tensor(
                current_features,
                dtype=torch.float32
            ),

            torch.tensor(
                normalized_delta,
                dtype=torch.float32
            ),

            torch.tensor(
                sample_weight,
                dtype=torch.float32
            )
        )


# ============================================================
# DATASETS
# ============================================================

train_dataset = PendulumDataset(
    trajectories,
    train_indices,
    SEQUENCE_LENGTH,
    SAMPLES_PER_EPOCH,
    feature_mean,
    feature_std,
    delta_mean,
    delta_std,
    ftle_weights,
    SEED
)


validation_dataset = PendulumDataset(
    trajectories,
    validation_indices,
    SEQUENCE_LENGTH,
    VALIDATION_SAMPLES,
    feature_mean,
    feature_std,
    delta_mean,
    delta_std,
    ftle_weights,
    SEED + 1
)


# ============================================================
# DATA LOADERS
# ============================================================

if (
    DEVICE.type == "cuda"
    and os.name != "nt"
):

    num_workers = min(
        4,
        os.cpu_count() or 1
    )

    pin_memory = True

else:

    num_workers = 0

    pin_memory = (
        DEVICE.type == "cuda"
    )


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(
        num_workers > 0
    )
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(
        num_workers > 0
    )
)


# ============================================================
# MODEL
# ============================================================

class HighAccuracyPendulumLSTM(
    nn.Module
):

    def __init__(
        self,
        input_size=6,
        hidden_size=256,
        num_layers=3,
        dropout=0.1
    ):

        super().__init__()


        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )


        # ----------------------------------------------------
        # Current-state branch
        # ----------------------------------------------------

        self.current_branch = nn.Sequential(
            nn.Linear(
                6,
                128
            ),

            nn.GELU(),

            nn.Linear(
                128,
                128
            ),

            nn.GELU()
        )


        # ----------------------------------------------------
        # Prediction head
        # ----------------------------------------------------

        self.head = nn.Sequential(
            nn.Linear(
                hidden_size + 128,
                256
            ),

            nn.GELU(),

            nn.Linear(
                256,
                256
            ),

            nn.GELU(),

            nn.Linear(
                256,
                4
            )
        )


    def forward(
        self,
        sequence,
        current_state
    ):

        output, _ = self.lstm(
            sequence
        )


        lstm_output = (
            output[:, -1, :]
        )


        current_output = (
            self.current_branch(
                current_state
            )
        )


        combined = torch.cat(
            [
                lstm_output,
                current_output
            ],
            dim=1
        )


        return self.head(
            combined
        )


# ============================================================
# CREATE MODEL
# ============================================================

model = HighAccuracyPendulumLSTM(
    input_size=6,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(
    DEVICE
)


print(model)
print()


# ============================================================
# LOSS FUNCTIONS
# ============================================================

state_criterion = nn.SmoothL1Loss(
    beta=0.1,
    reduction="none"
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING-RATE SCHEDULER
# ============================================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        threshold=1e-4,
        min_lr=1e-6
    )
)

# ============================================================
# AMP
# ============================================================

use_amp = (
    DEVICE.type == "cuda"
)


if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


# ============================================================
# NORMALIZED TENSOR CONSTANTS
# ============================================================

delta_mean_tensor = torch.tensor(
    delta_mean,
    dtype=torch.float32,
    device=DEVICE
)


delta_std_tensor = torch.tensor(
    delta_std,
    dtype=torch.float32,
    device=DEVICE
)


feature_mean_tensor = torch.tensor(
    feature_mean,
    dtype=torch.float32,
    device=DEVICE
)


feature_std_tensor = torch.tensor(
    feature_std,
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# DENORMALIZE DELTA
# ============================================================

def denormalize_delta_tensor(
    delta
):

    return (
        delta
        * delta_std_tensor.to(
            dtype=delta.dtype
        )
        +
        delta_mean_tensor.to(
            dtype=delta.dtype
        )
    )


# ============================================================
# DIFFERENTIABLE CURRENT STATE RECOVERY
# ============================================================

def normalized_features_to_physical_state(
    normalized_features
):
    """
    Convert normalized:

        [sin(theta1),
         cos(theta1),
         sin(theta2),
         cos(theta2),
         omega1,
         omega2]

    back into:

        [theta1,
         theta2,
         omega1,
         omega2]
    """

    physical_features = (
        normalized_features
        *
        feature_std_tensor.to(
            dtype=normalized_features.dtype
        )
        +
        feature_mean_tensor.to(
            dtype=normalized_features.dtype
        )
    )


    sin1 = physical_features[:, 0]

    cos1 = physical_features[:, 1]

    sin2 = physical_features[:, 2]

    cos2 = physical_features[:, 3]

    omega1 = physical_features[:, 4]

    omega2 = physical_features[:, 5]


    theta1 = torch.atan2(
        sin1,
        cos1
    )

    theta2 = torch.atan2(
        sin2,
        cos2
    )


    return torch.stack(
        [
            theta1,
            theta2,
            omega1,
            omega2
        ],
        dim=1
    )


# ============================================================
# DIFFERENTIABLE XY CALCULATION
# ============================================================

def torch_state_to_xy(
    current_state_features,
    delta
):
    """
    current_state_features:
        normalized 6-feature representation

    delta:
        physical state delta

    Returns:
        predicted [x1, y1, x2, y2]
    """

    current_state = (
        normalized_features_to_physical_state(
            current_state_features
        )
    )


    theta1 = (
        current_state[:, 0]
        +
        delta[:, 0]
    )

    theta2 = (
        current_state[:, 1]
        +
        delta[:, 1]
    )

    x1 = (
        L1 * torch.sin(theta1)
    )

    y1 = (
        -L1 * torch.cos(theta1)
    )

    x2 = (
        x1
        +
        L2 * torch.sin(theta2)
    )


    y2 = (
        y1
        -
        L2 * torch.cos(theta2)
    )


    return torch.stack(
        [
            x1,
            y1,
            x2,
            y2
        ],
        dim=1
    )


# ============================================================
# DIFFERENTIABLE DOUBLE-PENDULUM RK4 STEP
# ============================================================

def double_pendulum_derivatives_torch(
    state
):
    """
    Differentiable double-pendulum equations of motion.

    State format:

        [theta1, theta2, omega1, omega2]

    Angles are measured from the downward vertical.

    This function is fully differentiable with respect to
    the input state, which allows gradients to flow through
    the RK4 physics calculation and back into the LSTM.
    """

    theta1 = state[:, 0]
    theta2 = state[:, 1]

    omega1 = state[:, 2]
    omega2 = state[:, 3]

    # --------------------------------------------------------
    # Difference in angles
    # --------------------------------------------------------

    delta = theta2 - theta1

    sin_delta = torch.sin(delta)
    cos_delta = torch.cos(delta)

    # --------------------------------------------------------
    # Double-pendulum equations of motion
    # --------------------------------------------------------

    denominator1 = (
        L1
        * (
            2.0 * m1
            + m2
            - m2 * torch.cos(2.0 * delta)
        )
    )

    denominator2 = (
        L2
        * (
            2.0 * m1
            + m2
            - m2 * torch.cos(2.0 * delta)
        )
    )

    alpha1 = (
        -g * (2.0 * m1 + m2) * torch.sin(theta1)
        - m2 * g * torch.sin(theta1 - 2.0 * theta2)
        - 2.0 * sin_delta * m2
        * (
            omega2 ** 2 * L2
            + omega1 ** 2 * L1 * cos_delta
        )
    ) / denominator1

    alpha2 = (
        2.0
        * sin_delta
        * (
            omega1 ** 2 * L1 * (m1 + m2)
            + g * (m1 + m2) * torch.cos(theta1)
            + omega2 ** 2 * L2 * m2 * cos_delta
        )
    ) / denominator2

    return torch.stack(
        [
            omega1,
            omega2,
            alpha1,
            alpha2
        ],
        dim=1
    )


# ============================================================
# DIFFERENTIABLE RK4 PHYSICS STEP
# ============================================================

def differentiable_rk4_step(
    state,
    step_size
):
    """
    Performs one fully differentiable RK4 integration step.

    This is the physics model used by the physics-informed
    loss.

    Gradients can pass through every RK4 operation:

        LSTM prediction
              ↓
        physics loss
              ↓
        RK4 equations
              ↓
        predicted state
              ↓
        backpropagation
    """

    k1 = double_pendulum_derivatives_torch(
        state
    )

    k2 = double_pendulum_derivatives_torch(
        state
        + 0.5 * step_size * k1
    )

    k3 = double_pendulum_derivatives_torch(
        state
        + 0.5 * step_size * k2
    )

    k4 = double_pendulum_derivatives_torch(
        state
        + step_size * k3
    )

    next_state = (
        state
        + (
            step_size / 6.0
        ) * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )
    )

    return next_state


# ============================================================
# PHYSICS LOSS
# ============================================================

def calculate_physics_loss(
    current_features,
    predicted_delta
):
    """
    Physics-informed loss using a fully differentiable RK4
    double-pendulum physics step.

    The neural network predicts:

        predicted_state
        =
        current_state + predicted_delta

    The physics model independently predicts:

        physics_state
        =
        RK4(current_state, dt)

    The physics loss measures the difference between these
    two states.

    This replaces the old finite-difference acceleration
    calculation:

        (predicted_omega - omega) / dt

    with the actual double-pendulum equations of motion.

    Everything is differentiable with respect to
    predicted_delta.
    """

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    current_features = current_features.float()
    predicted_delta = predicted_delta.float()

    # --------------------------------------------------------
    # Convert normalized network input back into physical
    # double-pendulum state.
    #
    # State:
    #
    # [theta1, theta2, omega1, omega2]
    # --------------------------------------------------------

    current_state = (
        normalized_features_to_physical_state(
            current_features
        )
    )

    current_state = current_state.float()

    # --------------------------------------------------------
    # Neural-network prediction
    # --------------------------------------------------------

    predicted_state = (
        current_state
        +
        predicted_delta
    )

    # --------------------------------------------------------
    # Physics prediction
    #
    # IMPORTANT:
    #
    # This is a completely differentiable RK4 calculation.
    #
    # We intentionally DO NOT detach current_state or the
    # physics prediction from the computational graph.
    # --------------------------------------------------------

    physics_state = (
        differentiable_rk4_step(
            current_state,
            dt
        )
    )

    # --------------------------------------------------------
    # State residual
    #
    # Compare the AI's prediction with the RK4 physics
    # prediction.
    # --------------------------------------------------------

    state_residual = (
        predicted_state
        -
        physics_state
    )

    # --------------------------------------------------------
    # Normalize the four state components so that the
    # angular-position and angular-velocity errors have
    # comparable influence.
    # --------------------------------------------------------

    theta_scale = 1.0

    omega_scale = torch.sqrt(
        torch.tensor(
            g / max(
                min(L1, L2),
                1e-8
            ),
            dtype=torch.float32,
            device=current_state.device
        )
    )

    omega_scale = torch.clamp(
        omega_scale,
        min=1.0
    )

    # --------------------------------------------------------
    # Individual normalized residuals
    # --------------------------------------------------------

    theta1_residual = (
        state_residual[:, 0]
        / theta_scale
    )

    theta2_residual = (
        state_residual[:, 1]
        / theta_scale
    )

    omega1_residual = (
        state_residual[:, 2]
        / omega_scale
    )

    omega2_residual = (
        state_residual[:, 3]
        / omega_scale
    )

    # --------------------------------------------------------
    # Mean squared physics error
    # --------------------------------------------------------

    physics_loss = (
        theta1_residual ** 2
        +
        theta2_residual ** 2
        +
        omega1_residual ** 2
        +
        omega2_residual ** 2
    )

    # --------------------------------------------------------
    # Average over the batch
    # --------------------------------------------------------

    return torch.mean(
        physics_loss
    )

# ============================================================
# ENERGY LOSS
# ============================================================

def calculate_energy_loss(
    current_features,
    predicted_delta
):
    """
    Penalize changes in mechanical energy.

    The exact true trajectory has tiny numerical energy drift
    from RK4, but it should remain approximately conserved.

    We therefore use a normalized relative energy error.
    """

    current_features = (
        current_features.float()
    )

    predicted_delta = (
        predicted_delta.float()
    )


    current_state = (
        normalized_features_to_physical_state(
            current_features
        )
    )


    theta1 = current_state[:, 0]

    theta2 = current_state[:, 1]

    omega1 = current_state[:, 2]

    omega2 = current_state[:, 3]


    # --------------------------------------------------------
    # Current energy
    # --------------------------------------------------------

    current_energy = torch_energy(
        theta1,
        theta2,
        omega1,
        omega2
    )


    # --------------------------------------------------------
    # Predicted state
    # --------------------------------------------------------

    predicted_theta1 = (
        theta1
        +
        predicted_delta[:, 0]
    )

    predicted_theta2 = (
        theta2
        +
        predicted_delta[:, 1]
    )

    predicted_omega1 = (
        omega1
        +
        predicted_delta[:, 2]
    )

    predicted_omega2 = (
        omega2
        +
        predicted_delta[:, 3]
    )


    predicted_energy = torch_energy(
        predicted_theta1,
        predicted_theta2,
        predicted_omega1,
        predicted_omega2
    )


    # --------------------------------------------------------
    # Characteristic energy scale
    # --------------------------------------------------------

    energy_scale = (
        m1
        * g
        * L1
        +
        m2
        * g
        * (
            L1
            +
            L2
        )
        +
        1.0
    )


    # --------------------------------------------------------
    # Relative energy residual
    # --------------------------------------------------------

    energy_residual = (
        predicted_energy
        -
        current_energy
    ) / energy_scale


    return torch.mean(
        energy_residual
        ** 2
    )


# ============================================================
# COMPLETE LOSS
# ============================================================

def calculate_total_loss(
    current_features,
    predicted_delta,
    target_delta,
    sample_weights
):
    """Single-step physics-informed loss used for stable, fast training."""

    # --------------------------------------------------------
    # Make physics calculations float32.
    # --------------------------------------------------------

    predicted_delta_f32 = (
        predicted_delta.float()
    )

    target_delta_f32 = (
        target_delta.float()
    )


    # --------------------------------------------------------
    # State loss
    # --------------------------------------------------------

    state_loss_per_element = (
        state_criterion(
            predicted_delta_f32,
            target_delta_f32
        )
    )

    state_loss_per_sample = (
        torch.mean(
            state_loss_per_element,
            dim=1
        )
    )


    # --------------------------------------------------------
    # True physical delta
    # --------------------------------------------------------

    true_physical_delta = (
        denormalize_delta_tensor(
            target_delta_f32
        )
    )


    # --------------------------------------------------------
    # Predicted physical delta
    # --------------------------------------------------------

    predicted_physical_delta = (
        denormalize_delta_tensor(
            predicted_delta_f32
        )
    )


    # --------------------------------------------------------
    # Position loss
    # --------------------------------------------------------

    predicted_xy = (
        torch_state_to_xy(
            current_features,
            predicted_physical_delta
        )
    )

    target_xy = (
        torch_state_to_xy(
            current_features,
            true_physical_delta
        )
    )


    position_loss_per_sample = (
        torch.mean(
            (
                predicted_xy
                -
                target_xy
            )
            ** 2,
            dim=1
        )
    )


    # --------------------------------------------------------
    # Physics loss
    # --------------------------------------------------------

    # This is calculated as one batch-wide value.
    physics_loss = (
        calculate_physics_loss(
            current_features,
            predicted_physical_delta
        )
    )


    # --------------------------------------------------------
    # Energy loss
    # --------------------------------------------------------

    energy_loss = (
        calculate_energy_loss(
            current_features,
            predicted_physical_delta
        )
    )


    # --------------------------------------------------------
    # Base sample loss
    # --------------------------------------------------------

    sample_loss = (
        STATE_LOSS_WEIGHT
        * state_loss_per_sample
        +
        POSITION_LOSS_WEIGHT
        * position_loss_per_sample
    )


    # --------------------------------------------------------
    # Apply FTLE weighting to supervised losses.
    # --------------------------------------------------------

    sample_weights = (
        sample_weights.float()
    )


    sample_weights = (
        sample_weights
        /
        torch.mean(
            sample_weights
        )
    )


    weighted_supervised_loss = (
        torch.mean(
            sample_weights
            * sample_loss
        )
    )


    # --------------------------------------------------------
    # Physics and energy constraints.
    #
    # These are already averages over the batch.
    # --------------------------------------------------------

    total_loss = (
        weighted_supervised_loss
        +
        PHYSICS_LOSS_WEIGHT
        * physics_loss
        +
        ENERGY_LOSS_WEIGHT
        * energy_loss
    )


    return (
        total_loss,
        torch.mean(
            state_loss_per_sample
        ),
        torch.mean(
            position_loss_per_sample
        ),
        physics_loss,
        energy_loss
    )


# ============================================================
# TRAINING
# ============================================================

best_validation_loss = float(
    "inf"
)

epochs_without_improvement = 0

training_start = (
    time.perf_counter()
)


print("=" * 75)
print("Training the double pendulum model")
print("=" * 75)
print()

print(
    "The loss combines state accuracy, position accuracy, physics consistency, and energy conservation."
)

print(
    "More chaotic trajectories receive slightly more influence during training."
)

print()


for epoch in range(
    EPOCHS
):

    epoch_start = (
        time.perf_counter()
    )

    # ========================================================
    # TRAINING
    # ========================================================

    model.train()


    total_train_loss = 0.0

    total_train_state_loss = 0.0

    total_train_position_loss = 0.0

    total_train_physics_loss = 0.0

    total_train_energy_loss = 0.0

    train_batches = 0


    for (
        X,
        current,
        target_delta,
        sample_weights
    ) in train_loader:


        X = X.to(
            DEVICE,
            non_blocking=True
        )

        current = current.to(
            DEVICE,
            non_blocking=True
        )

        target_delta = target_delta.to(
            DEVICE,
            non_blocking=True
        )

        sample_weights = sample_weights.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        if use_amp:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                predicted_delta = model(
                    X,
                    current
                )


        else:

            predicted_delta = model(
                X,
                current
            )


        # ----------------------------------------------------
        # Physics-informed total loss
        # ----------------------------------------------------

        (
            loss,
            state_loss,
            position_loss,
            physics_loss,
            energy_loss
        ) = calculate_total_loss(
            current,
            predicted_delta,
            target_delta,
            sample_weights
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        if use_amp:

            scaler.scale(
                loss
            ).backward()


            scaler.unscale_(
                optimizer
            )


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )


            scaler.step(
                optimizer
            )


            scaler.update()


        else:

            loss.backward()


            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )


            optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_train_loss += (
            loss.item()
        )

        total_train_state_loss += (
            state_loss.item()
        )

        total_train_position_loss += (
            position_loss.item()
        )

        total_train_physics_loss += (
            physics_loss.item()
        )

        total_train_energy_loss += (
            energy_loss.item()
        )

        train_batches += 1


    train_loss = (
        total_train_loss
        /
        train_batches
    )


    train_state_loss = (
        total_train_state_loss
        /
        train_batches
    )


    train_position_loss = (
        total_train_position_loss
        /
        train_batches
    )


    train_physics_loss = (
        total_train_physics_loss
        /
        train_batches
    )


    train_energy_loss = (
        total_train_energy_loss
        /
        train_batches
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    total_validation_loss = 0.0

    total_validation_state_loss = 0.0

    total_validation_position_loss = 0.0

    total_validation_physics_loss = 0.0

    total_validation_energy_loss = 0.0

    validation_batches = 0


    with torch.no_grad():

        for (
            X,
            current,
            target_delta,
            sample_weights
        ) in validation_loader:


            X = X.to(
                DEVICE,
                non_blocking=True
            )

            current = current.to(
                DEVICE,
                non_blocking=True
            )

            target_delta = target_delta.to(
                DEVICE,
                non_blocking=True
            )

            sample_weights = sample_weights.to(
                DEVICE,
                non_blocking=True
            )


            if use_amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    predicted_delta = model(
                        X,
                        current
                    )

            else:

                predicted_delta = model(
                    X,
                    current
                )


            (
                loss,
                state_loss,
                position_loss,
                physics_loss,
                energy_loss
            ) = calculate_total_loss(
                current,
                predicted_delta,
                target_delta,
                sample_weights
            )

            total_validation_loss += (
                loss.item()
            )

            total_validation_state_loss += (
                state_loss.item()
            )

            total_validation_position_loss += (
                position_loss.item()
            )

            total_validation_physics_loss += (
                physics_loss.item()
            )

            total_validation_energy_loss += (
                energy_loss.item()
            )

            validation_batches += 1


    validation_loss = (
        total_validation_loss
        /
        validation_batches
    )


    validation_state_loss = (
        total_validation_state_loss
        /
        validation_batches
    )


    validation_position_loss = (
        total_validation_position_loss
        /
        validation_batches
    )


    validation_physics_loss = (
        total_validation_physics_loss
        /
        validation_batches
    )


    validation_energy_loss = (
        total_validation_energy_loss
        /
        validation_batches
    )


    # ========================================================
    # LEARNING-RATE SCHEDULE
    # ========================================================

    scheduler.step(validation_loss)


    epoch_time = (
        time.perf_counter()
        -
        epoch_start
    )


    lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    # ========================================================
    # REPORT
    # ========================================================

    print(
        f"Epoch {epoch + 1} of {EPOCHS}: training loss is {train_loss:.6e}, "
        f"validation loss is {validation_loss:.6e}, state loss is {train_state_loss:.3e}, "
        f"position loss is {train_position_loss:.3e}, physics loss is {train_physics_loss:.3e}, "
        f"energy loss is {train_energy_loss:.3e}, learning rate is {lr:.3e}, "
        f"and the elapsed time is {epoch_time:.1f} seconds."
    )


    print(
        f"Validation breakdown: state loss is {validation_state_loss:.3e}, "
        f"position loss is {validation_position_loss:.3e}, physics loss is {validation_physics_loss:.3e}, "
        f"and energy loss is {validation_energy_loss:.3e}."
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if (
        validation_loss
        <
        best_validation_loss
    ):

        best_validation_loss = (
            validation_loss
        )

        epochs_without_improvement = 0


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "feature_mean":
                    feature_mean,

                "feature_std":
                    feature_std,

                "delta_mean":
                    delta_mean,

                "delta_std":
                    delta_std,

                "m1":
                    m1,

                "m2":
                    m2,

                "L1":
                    L1,

                "L2":
                    L2,

                "g":
                    g,

                "dt":
                    dt,

                "sequence_length":
                    SEQUENCE_LENGTH,

                "hidden_size":
                    HIDDEN_SIZE,

                "num_layers":
                    NUM_LAYERS,

                "dropout":
                    DROPOUT,

                "state_loss_weight":
                    STATE_LOSS_WEIGHT,

                "position_loss_weight":
                    POSITION_LOSS_WEIGHT,

                "physics_loss_weight":
                    PHYSICS_LOSS_WEIGHT,

                "energy_loss_weight":
                    ENERGY_LOSS_WEIGHT,

                "ftle_weight_min":
                    FTLE_WEIGHT_MIN,

                "ftle_weight_max":
                    FTLE_WEIGHT_MAX,

                "validation_loss":
                    validation_loss
            },
            MODEL_PATH
        )


        print(
            "    BEST PILSTM MODEL SAVED"
        )


    else:

        epochs_without_improvement += 1


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()

        print(
            "Early stopping."
        )

        break


# ============================================================
# TRAINING TIME
# ============================================================

training_time = (
    time.perf_counter()
    -
    training_start
)


print()

print(
    f"Training time: "
    f"{training_time / 60:.2f} minutes"
)

print()


# ============================================================
# LOAD BEST MODEL
# ============================================================

try:

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

except TypeError:

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model.eval()


# ============================================================
# SINGLE-STEP PREDICTION
# ============================================================

def predict_next_state(
    state_sequence
):
    """
    Predict exactly one timestep forward.

    state_sequence shape:

        (SEQUENCE_LENGTH, 4)
    """

    if len(state_sequence) != SEQUENCE_LENGTH:

        raise ValueError(
            f"Expected exactly "
            f"{SEQUENCE_LENGTH} states, "
            f"got {len(state_sequence)}."
        )


    # --------------------------------------------------------
    # Convert states to features
    # --------------------------------------------------------

    features = state_to_features(
        state_sequence
    )


    features = (
        features
        -
        feature_mean
    ) / feature_std


    current_features = (
        features[-1]
    )


    # --------------------------------------------------------
    # Torch tensors
    # --------------------------------------------------------

    X = torch.tensor(
        features,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)


    current = torch.tensor(
        current_features,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)


    # --------------------------------------------------------
    # Neural-network prediction
    # --------------------------------------------------------

    with torch.no_grad():

        predicted_normalized_delta = (
            model(
                X,
                current
            )
        )


    predicted_normalized_delta = (
        predicted_normalized_delta
        .cpu()
        .numpy()[0]
    )


    # --------------------------------------------------------
    # Convert back to physical delta
    # --------------------------------------------------------

    predicted_delta = (
        predicted_normalized_delta
        *
        delta_std
        +
        delta_mean
    )


    # --------------------------------------------------------
    # Apply delta
    # --------------------------------------------------------

    current_state = (
        state_sequence[-1]
        .astype(
            np.float64
        )
    )


    predicted_state = apply_delta(
        current_state,
        predicted_delta
    )


    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not np.all(
        np.isfinite(
            predicted_state
        )
    ):

        raise FloatingPointError(
            "PILSTM produced NaN or infinity."
        )


    return predicted_state


# ============================================================
# AUTOREGRESSIVE PREDICTION
# ============================================================

def predict_future(
    initial_sequence,
    steps
):
    """
    Recursively predict future states.

    The PILSTM receives its own previous prediction,
    exactly as it will during the Pygame simulation.
    """

    sequence = (
        initial_sequence
        .copy()
        .astype(
            np.float64
        )
    )


    if len(sequence) != SEQUENCE_LENGTH:

        raise ValueError(
            f"Expected initial sequence "
            f"of length {SEQUENCE_LENGTH}."
        )


    predictions = []


    for _ in range(
        steps
    ):

        next_state = (
            predict_next_state(
                sequence
            )
        )


        predictions.append(
            next_state
        )


        sequence = np.vstack(
            [
                sequence[1:],
                next_state
            ]
        )


    return np.asarray(
        predictions
    )


print(
    "The trained model is ready to use."
)

print(
    f"The best validation loss is {best_validation_loss:.6e}."
)

print()






















































































































































































































































































































































































































































































































#add-ons:


# Required variables/functions already created above:
#
#   model
#   predict_next_state()
#   SEQUENCE_LENGTH
#   dt
#   m1
#   m2
#   L1
#   L2
#   g
#   DEVICE
#
# Install:
#
#   pip install pygame
#
# ============================================================


import math
import numpy as np
import pygame


# ============================================================
# SETTINGS
# ============================================================

SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 850

FPS = 60

# Physics steps performed per displayed frame.
#
# Keep this LOW because every AI prediction requires an LSTM
# forward pass.
SIMULATION_STEPS = 1

MIN_SIMULATION_STEPS = 1
MAX_SIMULATION_STEPS = 20

# ------------------------------------------------------------
# Initial condition
# ------------------------------------------------------------

INITIAL_THETA1 = math.radians(120.0)
INITIAL_THETA2 = math.radians(-40.0)

INITIAL_W1 = 0.0
INITIAL_W2 = 0.0

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

PREDICTION_TOLERANCE = 1e-3

# Maximum number of graph points.
MAX_GRAPH_POINTS = 1500

# Maximum number of trajectory points.
MAX_TRAIL_POINTS = 1500

# ------------------------------------------------------------
# Sensitivity
# ------------------------------------------------------------

SENSITIVITY_EPSILON = 1e-7

# Use full state for the Lyapunov-style sensitivity measure.
# This is much more meaningful than measuring only x/y.
OMEGA_SCALE = math.sqrt(L1 / g)


# ============================================================
# COLORS
# ============================================================

BLACK = (8, 8, 12)
WHITE = (240, 240, 245)
GRAY = (135, 135, 145)
DARK_GRAY = (28, 28, 38)

BLUE = (60, 145, 255)
RED = (245, 70, 70)
GREEN = (70, 220, 120)
ORANGE = (255, 160, 60)
YELLOW = (245, 220, 70)
CYAN = (70, 220, 230)


# ============================================================
# PYGAME
# ============================================================

pygame.init()

screen = pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT)
)

pygame.display.set_caption(
    "Double Pendulum — AI Chaos Lab"
)

clock = pygame.time.Clock()


# ============================================================
# FONTS
# ============================================================

FONT_SMALL = pygame.font.SysFont(
    "consolas",
    15
)

FONT_MEDIUM = pygame.font.SysFont(
    "consolas",
    19
)

FONT_LARGE = pygame.font.SysFont(
    "consolas",
    25,
    bold=True
)

FONT_TITLE = pygame.font.SysFont(
    "consolas",
    30,
    bold=True
)


# ============================================================
# SCREEN GEOMETRY
# ============================================================

WORLD_SCALE = 210.0

ORIGIN_X = 410
ORIGIN_Y = 350

GRAPH_X = 25
GRAPH_Y = 595
GRAPH_WIDTH = 850
GRAPH_HEIGHT = 225

PANEL_X = 925
PANEL_Y = 65
PANEL_WIDTH = 450
PANEL_HEIGHT = 755


# ============================================================
# ANGLE NORMALIZATION
# ============================================================

def wrap_angle(angle):

    return (
        (angle + np.pi)
        % (2.0 * np.pi)
        - np.pi
    )


# ============================================================
# CORRECT DOUBLE-PENDULUM PHYSICS
# ============================================================
#
# IMPORTANT:
#
# This uses the SAME mass-matrix formulation as the dataset
# generator above.
#
# That eliminates a second, slightly different physics model
# from being used by the visualization.
# ============================================================

def physics_derivatives(state):

    theta1 = float(state[0])
    theta2 = float(state[1])

    omega1 = float(state[2])
    omega2 = float(state[3])

    delta = theta2 - theta1

    sin_delta = math.sin(delta)
    cos_delta = math.cos(delta)

    A = (
        (m1 + m2)
        * L1 ** 2
    )

    B = (
        m2
        * L1
        * L2
        * cos_delta
    )

    C = (
        m2
        * L2 ** 2
    )

    rhs1 = (
        m2
        * L1
        * L2
        * omega2 ** 2
        * sin_delta

        - (m1 + m2)
        * g
        * L1
        * math.sin(theta1)
    )

    rhs2 = (
        -m2
        * L1
        * L2
        * omega1 ** 2
        * sin_delta

        -m2
        * g
        * L2
        * math.sin(theta2)
    )

    determinant = (
        A * C
        - B ** 2
    )

    # Safety against numerical singularity.
    if abs(determinant) < 1e-14:

        raise FloatingPointError(
            "Double-pendulum mass matrix became "
            "numerically singular."
        )

    alpha1 = (
        rhs1 * C
        - B * rhs2
    ) / determinant

    alpha2 = (
        A * rhs2
        - B * rhs1
    ) / determinant

    return np.array(
        [
            omega1,
            omega2,
            alpha1,
            alpha2
        ],
        dtype=np.float64
    )


# ============================================================
# RK4
# ============================================================

def rk4_step(
    state,
    timestep
):

    state = np.asarray(
        state,
        dtype=np.float64
    )

    k1 = physics_derivatives(
        state
    )

    k2 = physics_derivatives(
        state
        + 0.5 * timestep * k1
    )

    k3 = physics_derivatives(
        state
        + 0.5 * timestep * k2
    )

    k4 = physics_derivatives(
        state
        + timestep * k3
    )

    result = (
        state
        + timestep / 6.0
        * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )
    )

    result[0] = wrap_angle(
        result[0]
    )

    result[1] = wrap_angle(
        result[1]
    )

    return result


# ============================================================
# POSITION
# ============================================================

def state_to_xy(
    state
):

    theta1 = float(state[0])
    theta2 = float(state[1])

    x1 = (
        L1
        * math.sin(theta1)
    )

    y1 = (
        -L1
        * math.cos(theta1)
    )

    x2 = (
        x1
        + L2
        * math.sin(theta2)
    )

    y2 = (
        y1
        - L2
        * math.cos(theta2)
    )

    return np.array(
        [
            x2,
            y2
        ],
        dtype=np.float64
    )


def state_to_positions(
    state
):

    theta1 = float(state[0])
    theta2 = float(state[1])

    x1 = (
        L1
        * math.sin(theta1)
    )

    y1 = (
        -L1
        * math.cos(theta1)
    )

    x2 = (
        x1
        + L2
        * math.sin(theta2)
    )

    y2 = (
        y1
        - L2
        * math.cos(theta2)
    )

    return (
        (0.0, 0.0),
        (x1, y1),
        (x2, y2)
    )


def world_to_screen(
    position
):

    x, y = position

    return (
        int(
            ORIGIN_X
            + x * WORLD_SCALE
        ),
        int(
            ORIGIN_Y
            - y * WORLD_SCALE
        )
    )


# ============================================================
# ENERGY
# ============================================================

def calculate_energy(
    state
):

    theta1 = float(state[0])
    theta2 = float(state[1])

    omega1 = float(state[2])
    omega2 = float(state[3])

    kinetic1 = (
        0.5
        * m1
        * L1 ** 2
        * omega1 ** 2
    )

    kinetic2 = (
        0.5
        * m2
        * (
            L1 ** 2
            * omega1 ** 2

            + L2 ** 2
            * omega2 ** 2

            + 2.0
            * L1
            * L2
            * omega1
            * omega2
            * math.cos(
                theta1 - theta2
            )
        )
    )

    y1 = (
        -L1
        * math.cos(theta1)
    )

    y2 = (
        y1
        - L2
        * math.cos(theta2)
    )

    potential1 = (
        m1
        * g
        * y1
    )

    potential2 = (
        m2
        * g
        * y2
    )

    return (
        kinetic1
        + kinetic2
        + potential1
        + potential2
    )


# ============================================================
# STATE DISTANCE
# ============================================================

def state_distance(
    state_a,
    state_b
):

    difference = (
        np.asarray(state_a)
        - np.asarray(state_b)
    ).astype(np.float64)

    # Angular velocities have different units, so scale them
    # to the characteristic pendulum time.
    difference = difference.copy()

    difference[0] = wrap_angle(
        difference[0]
    )

    difference[1] = wrap_angle(
        difference[1]
    )

    difference[2:] *= OMEGA_SCALE

    return float(
        np.linalg.norm(
            difference
        )
    )


def xy_distance(
    state_a,
    state_b
):

    return float(
        np.linalg.norm(
            state_to_xy(state_a)
            - state_to_xy(state_b)
        )
    )


# ============================================================
# FINITE-TIME LYAPUNOV ESTIMATE
# ============================================================

def finite_time_lambda(
    initial_distance,
    current_distance,
    elapsed
):

    if (
        elapsed <= 0.0
        or initial_distance <= 0.0
        or current_distance <= 0.0
    ):

        return 0.0

    return (
        math.log(
            current_distance
            / initial_distance
        )
        / elapsed
    )


# ============================================================
# INITIAL STATE
# ============================================================

def make_initial_state():

    return np.array(
        [
            INITIAL_THETA1,
            INITIAL_THETA2,
            INITIAL_W1,
            INITIAL_W2
        ],
        dtype=np.float64
    )


# ============================================================
# AI HISTORY
# ============================================================

def make_initial_history(
    initial_state
):

    history = []

    state = (
        initial_state.copy()
    )

    history.append(
        state.copy()
    )

    for _ in range(
        SEQUENCE_LENGTH - 1
    ):

        state = rk4_step(
            state,
            dt
        )

        history.append(
            state.copy()
        )

    return [
        np.asarray(
            x,
            dtype=np.float64
        )
        for x in history
    ]


# ============================================================
# AI PREDICTION
# ============================================================

def ai_predict(
    history
):

    if len(history) < SEQUENCE_LENGTH:

        raise RuntimeError(
            "AI history is shorter than "
            "SEQUENCE_LENGTH."
        )

    sequence = np.asarray(
        history[
            -SEQUENCE_LENGTH:
        ],
        dtype=np.float64
    )

    prediction = (
        predict_next_state(
            sequence
        )
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64
    )

    prediction = np.squeeze(
        prediction
    )

    if prediction.size != 4:

        raise ValueError(
            "predict_next_state() returned "
            f"shape {prediction.shape}. "
            "Expected exactly four state values."
        )

    prediction = prediction.reshape(
        4
    )

    prediction[0] = wrap_angle(
        prediction[0]
    )

    prediction[1] = wrap_angle(
        prediction[1]
    )

    if not np.all(
        np.isfinite(prediction)
    ):

        raise FloatingPointError(
            "AI produced NaN or infinite values."
        )

    return prediction


# ============================================================
# RESET
# ============================================================

def reset_simulation():

    global actual_state
    global ai_state

    global actual_history
    global ai_history

    global sensitivity_state_a
    global sensitivity_state_b

    global sensitivity_initial_distance

    global actual_trail
    global ai_trail

    global error_history
    global error_time_history

    global elapsed_time

    global prediction_horizon

    global last_error

    global initial_energy

    global ai_failed

    # --------------------------------------------------------
    # Physical state
    # --------------------------------------------------------

    initial = (
        make_initial_state()
    )

    # Warm up the history so that the LSTM has the exact
    # sequence length it expects.
    actual_history = (
        make_initial_history(
            initial
        )
    )

    actual_state = (
        actual_history[-1].copy()
    )

    # --------------------------------------------------------
    # AI starts EXACTLY from the same state/history.
    # --------------------------------------------------------

    ai_history = [
        state.copy()
        for state in actual_history
    ]

    ai_state = (
        ai_history[-1].copy()
    )

    # --------------------------------------------------------
    # Sensitivity pair
    # --------------------------------------------------------

    sensitivity_state_a = (
        actual_state.copy()
    )

    sensitivity_state_b = (
        actual_state.copy()
    )

    sensitivity_state_b[0] += (
        SENSITIVITY_EPSILON
    )

    sensitivity_initial_distance = (
        state_distance(
            sensitivity_state_a,
            sensitivity_state_b
        )
    )

    # --------------------------------------------------------
    # Trails
    # --------------------------------------------------------

    actual_trail = [
        state_to_xy(
            state
        )
        for state in actual_history
    ]

    ai_trail = [
        state_to_xy(
            state
        )
        for state in ai_history
    ]

    # --------------------------------------------------------
    # Graphs
    # --------------------------------------------------------

    error_history = []

    error_time_history = []

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    elapsed_time = 0.0

    prediction_horizon = None

    last_error = 0.0

    # --------------------------------------------------------
    # Energy
    # --------------------------------------------------------

    initial_energy = (
        calculate_energy(
            actual_state
        )
    )

    ai_failed = False


# ============================================================
# GLOBAL STATE
# ============================================================

actual_state = make_initial_state()
ai_state = actual_state.copy()

actual_history = []
ai_history = []

sensitivity_state_a = actual_state.copy()
sensitivity_state_b = actual_state.copy()

sensitivity_initial_distance = 0.0

actual_trail = []
ai_trail = []

error_history = []
error_time_history = []

elapsed_time = 0.0

prediction_horizon = None

last_error = 0.0

initial_energy = 0.0

ai_enabled = True
sensitivity_enabled = True
trails_enabled = True
paused = False

ai_failed = False

simulation_steps = SIMULATION_STEPS


# ============================================================
# RESET
# ============================================================

reset_simulation()


# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(
    text,
    x,
    y,
    color=WHITE,
    font=FONT_SMALL
):

    surface = font.render(
        str(text),
        True,
        color
    )

    screen.blit(
        surface,
        (
            x,
            y
        )
    )


# ============================================================
# DRAW PENDULUM
# ============================================================

def draw_pendulum(
    state,
    rod_color,
    mass_color,
    trail,
    show_trail
):

    pivot, mass1, mass2 = (
        state_to_positions(
            state
        )
    )

    pivot_screen = (
        world_to_screen(
            pivot
        )
    )

    mass1_screen = (
        world_to_screen(
            mass1
        )
    )

    mass2_screen = (
        world_to_screen(
            mass2
        )
    )

    # --------------------------------------------------------
    # Trail
    # --------------------------------------------------------

    if (
        show_trail
        and len(trail) > 1
    ):

        points = [
            world_to_screen(
                point
            )
            for point in trail
        ]

        pygame.draw.lines(
            screen,
            rod_color,
            False,
            points,
            2
        )

    # --------------------------------------------------------
    # Rods
    # --------------------------------------------------------

    pygame.draw.line(
        screen,
        rod_color,
        pivot_screen,
        mass1_screen,
        5
    )

    pygame.draw.line(
        screen,
        rod_color,
        mass1_screen,
        mass2_screen,
        5
    )

    # --------------------------------------------------------
    # Pivot
    # --------------------------------------------------------

    pygame.draw.circle(
        screen,
        WHITE,
        pivot_screen,
        8
    )

    # --------------------------------------------------------
    # Masses
    # --------------------------------------------------------

    pygame.draw.circle(
        screen,
        mass_color,
        mass1_screen,
        13
    )

    pygame.draw.circle(
        screen,
        mass_color,
        mass2_screen,
        16
    )


# ============================================================
# ERROR GRAPH
# ============================================================

def draw_error_graph():

    rect = (
        GRAPH_X,
        GRAPH_Y,
        GRAPH_WIDTH,
        GRAPH_HEIGHT
    )

    pygame.draw.rect(
        screen,
        DARK_GRAY,
        rect
    )

    pygame.draw.rect(
        screen,
        GRAY,
        rect,
        1
    )

    draw_text(
        "LSTM ENDPOINT POSITION ERROR",
        GRAPH_X + 10,
        GRAPH_Y + 8,
        WHITE,
        FONT_MEDIUM
    )

    if len(error_history) < 2:

        return

    maximum = max(
        max(error_history),
        PREDICTION_TOLERANCE
    )

    maximum *= 1.2

    if maximum <= 0.0:

        maximum = 1.0

    usable_width = (
        GRAPH_WIDTH - 20
    )

    usable_height = (
        GRAPH_HEIGHT - 50
    )

    # --------------------------------------------------------
    # Tolerance
    # --------------------------------------------------------

    fraction = (
        PREDICTION_TOLERANCE
        / maximum
    )

    tolerance_y = (
        GRAPH_Y
        + GRAPH_HEIGHT
        - 15
        - int(
            fraction
            * usable_height
        )
    )

    pygame.draw.line(
        screen,
        YELLOW,
        (
            GRAPH_X,
            tolerance_y
        ),
        (
            GRAPH_X + GRAPH_WIDTH,
            tolerance_y
        ),
        2
    )

    # --------------------------------------------------------
    # Error curve
    # --------------------------------------------------------

    points = []

    for i, value in enumerate(
        error_history
    ):

        x_fraction = (
            i
            / max(
                len(error_history) - 1,
                1
            )
        )

        y_fraction = min(
            value / maximum,
            1.0
        )

        px = (
            GRAPH_X
            + 10
            + int(
                x_fraction
                * usable_width
            )
        )

        py = (
            GRAPH_Y
            + GRAPH_HEIGHT
            - 15
            - int(
                y_fraction
                * usable_height
            )
        )

        points.append(
            (
                px,
                py
            )
        )

    if len(points) >= 2:

        pygame.draw.lines(
            screen,
            RED,
            False,
            points,
            2
        )


# ============================================================
# INFORMATION PANEL
# ============================================================

def draw_panel():

    pygame.draw.rect(
        screen,
        DARK_GRAY,
        (
            PANEL_X,
            PANEL_Y,
            PANEL_WIDTH,
            PANEL_HEIGHT
        )
    )

    pygame.draw.rect(
        screen,
        GRAY,
        (
            PANEL_X,
            PANEL_Y,
            PANEL_WIDTH,
            PANEL_HEIGHT
        ),
        1
    )

    x = PANEL_X + 20
    y = PANEL_Y + 15

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    draw_text(
        "AI CHAOS LAB",
        x,
        y,
        WHITE,
        FONT_LARGE
    )

    y += 42

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    draw_text(
        "AI:",
        x,
        y,
        WHITE,
        FONT_MEDIUM
    )

    if ai_failed:

        ai_status = "FAILED"
        ai_color = RED

    elif ai_enabled:

        ai_status = "ON"
        ai_color = GREEN

    else:

        ai_status = "OFF"
        ai_color = GRAY

    draw_text(
        ai_status,
        x + 90,
        y,
        ai_color,
        FONT_MEDIUM
    )

    y += 28

    draw_text(
        "Simulation:",
        x,
        y
    )

    draw_text(
        "PAUSED"
        if paused
        else "RUNNING",
        x + 130,
        y,
        YELLOW
        if paused
        else GREEN
    )

    y += 28

    draw_text(
        "Sensitivity:",
        x,
        y
    )

    draw_text(
        "ON"
        if sensitivity_enabled
        else "OFF",
        x + 130,
        y,
        GREEN
        if sensitivity_enabled
        else GRAY
    )

    y += 35

    # --------------------------------------------------------
    # Physics
    # --------------------------------------------------------

    draw_text(
        "PHYSICS",
        x,
        y,
        CYAN,
        FONT_MEDIUM
    )

    y += 30

    draw_text(
        f"dt: {dt:.7g} s",
        x,
        y
    )

    y += 25

    draw_text(
        f"Steps/frame: {simulation_steps}",
        x,
        y
    )

    y += 25

    draw_text(
        f"Simulation time: {elapsed_time:.5f} s",
        x,
        y
    )

    y += 35

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    draw_text(
        "LSTM",
        x,
        y,
        RED,
        FONT_MEDIUM
    )

    y += 30

    draw_text(
        "Endpoint error:",
        x,
        y
    )

    draw_text(
        f"{last_error:.5e} m",
        x + 220,
        y,
        GREEN
        if last_error <= PREDICTION_TOLERANCE
        else RED
    )

    y += 27

    draw_text(
        "Tolerance:",
        x,
        y
    )

    draw_text(
        f"{PREDICTION_TOLERANCE:.1e} m",
        x + 220,
        y,
        YELLOW
    )

    y += 27

    draw_text(
        "Prediction horizon:",
        x,
        y
    )

    if prediction_horizon is None:

        horizon_text = "not reached"
        horizon_color = GREEN

    else:

        horizon_text = (
            f"{prediction_horizon:.5f} s"
        )

        horizon_color = YELLOW

    draw_text(
        horizon_text,
        x + 220,
        y,
        horizon_color
    )

    y += 38

    # --------------------------------------------------------
    # Sensitivity
    # --------------------------------------------------------

    draw_text(
        "CHAOS / SENSITIVITY",
        x,
        y,
        ORANGE,
        FONT_MEDIUM
    )

    y += 30

    current_distance = (
        state_distance(
            sensitivity_state_a,
            sensitivity_state_b
        )
    )

    amplification = (
        current_distance
        / max(
            sensitivity_initial_distance,
            1e-300
        )
    )

    lambda_value = (
        finite_time_lambda(
            sensitivity_initial_distance,
            current_distance,
            elapsed_time
        )
    )

    draw_text(
        "Initial state Δ:",
        x,
        y
    )

    draw_text(
        f"{sensitivity_initial_distance:.3e}",
        x + 220,
        y
    )

    y += 27

    draw_text(
        "Current state Δ:",
        x,
        y
    )

    draw_text(
        f"{current_distance:.3e}",
        x + 220,
        y,
        ORANGE
    )

    y += 27

    draw_text(
        "Amplification:",
        x,
        y
    )

    draw_text(
        f"{amplification:.3e}x",
        x + 220,
        y,
        ORANGE
    )

    y += 27

    draw_text(
        "FTLE:",
        x,
        y
    )

    draw_text(
        f"{lambda_value:.5e} /s",
        x + 220,
        y,
        ORANGE
    )

    y += 38

    # --------------------------------------------------------
    # Energy
    # --------------------------------------------------------

    current_energy = (
        calculate_energy(
            actual_state
        )
    )

    energy_error = (
        current_energy
        - initial_energy
    )

    relative_energy_error = (
        abs(energy_error)
        / max(
            abs(initial_energy),
            1e-12
        )
    )

    draw_text(
        "ENERGY CHECK",
        x,
        y,
        GREEN,
        FONT_MEDIUM
    )

    y += 30

    draw_text(
        "ΔE:",
        x,
        y
    )

    draw_text(
        f"{energy_error:.4e} J",
        x + 220,
        y,
        GREEN
    )

    y += 27

    draw_text(
        "Relative:",
        x,
        y
    )

    draw_text(
        f"{relative_energy_error:.3e}",
        x + 220,
        y,
        GREEN
    )

    y += 40

    # --------------------------------------------------------
    # Controls
    # --------------------------------------------------------

    draw_text(
        "CONTROLS",
        x,
        y,
        WHITE,
        FONT_MEDIUM
    )

    y += 28

    controls = [
        "SPACE  pause / resume",
        "A      toggle AI",
        "S      toggle sensitivity",
        "T      toggle trails",
        "R      reset",
        "UP     simulation speed +",
        "DOWN   simulation speed -",
        "ESC    quit"
    ]

    for text in controls:

        draw_text(
            text,
            x,
            y,
            GRAY
        )

        y += 21


# ============================================================
# MAIN LOOP
# ============================================================

running = True

while running:

    # ========================================================
    # EVENTS
    # ========================================================

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:

                running = False

            elif event.key == pygame.K_SPACE:

                paused = not paused

            elif event.key == pygame.K_a:

                if not ai_failed:

                    ai_enabled = (
                        not ai_enabled
                    )

            elif event.key == pygame.K_s:

                sensitivity_enabled = (
                    not sensitivity_enabled
                )

            elif event.key == pygame.K_t:

                trails_enabled = (
                    not trails_enabled
                )

            elif event.key == pygame.K_r:

                reset_simulation()

            elif event.key == pygame.K_UP:

                simulation_steps = min(
                    simulation_steps + 1,
                    MAX_SIMULATION_STEPS
                )

            elif event.key == pygame.K_DOWN:

                simulation_steps = max(
                    simulation_steps - 1,
                    MIN_SIMULATION_STEPS
                )


    # ========================================================
    # SIMULATION
    # ========================================================

    if not paused:

        for _ in range(
            simulation_steps
        ):

            # ------------------------------------------------
            # REAL PHYSICS
            # ------------------------------------------------

            actual_state = rk4_step(
                actual_state,
                dt
            )

            elapsed_time += dt

            actual_history.append(
                actual_state.copy()
            )

            if len(actual_history) > (
                SEQUENCE_LENGTH
            ):

                actual_history.pop(0)

            # ------------------------------------------------
            # ACTUAL TRAIL
            # ------------------------------------------------

            actual_trail.append(
                state_to_xy(
                    actual_state
                )
            )

            if len(actual_trail) > (
                MAX_TRAIL_POINTS
            ):

                actual_trail.pop(0)

            # ------------------------------------------------
            # AI
            # ------------------------------------------------
            #
            # IMPORTANT:
            #
            # The AI is recursive here.
            # It receives its own previous prediction.
            #
            # This is intentionally different from the
            # one-step test above.
            # ------------------------------------------------

            if ai_enabled:

                try:

                    ai_state = ai_predict(
                        ai_history
                    )

                    ai_history.append(
                        ai_state.copy()
                    )

                    if len(ai_history) > (
                        SEQUENCE_LENGTH
                    ):

                        ai_history.pop(0)

                    ai_trail.append(
                        state_to_xy(
                            ai_state
                        )
                    )

                    if len(ai_trail) > (
                        MAX_TRAIL_POINTS
                    ):

                        ai_trail.pop(0)

                    # ----------------------------------------
                    # Position error
                    # ----------------------------------------

                    last_error = (
                        xy_distance(
                            actual_state,
                            ai_state
                        )
                    )

                    error_history.append(
                        last_error
                    )

                    error_time_history.append(
                        elapsed_time
                    )

                    if len(error_history) > (
                        MAX_GRAPH_POINTS
                    ):

                        error_history.pop(0)

                        error_time_history.pop(0)

                    # ----------------------------------------
                    # Prediction horizon
                    # ----------------------------------------

                    if (
                        prediction_horizon is None
                        and last_error
                        > PREDICTION_TOLERANCE
                    ):

                        prediction_horizon = (
                            elapsed_time
                        )

                except Exception as error:

                    ai_failed = True
                    ai_enabled = False

                    print()
                    print("=" * 70)
                    print("AI PREDICTION ERROR")
                    print("=" * 70)
                    print(
                        repr(error)
                    )
                    print()
                    print(
                        "AI visualization has been disabled."
                    )
                    print(
                        "Press R to reset and try again."
                    )
                    print("=" * 70)

            # ------------------------------------------------
            # SENSITIVITY
            # ------------------------------------------------

            if sensitivity_enabled:

                sensitivity_state_a = rk4_step(
                    sensitivity_state_a,
                    dt
                )

                sensitivity_state_b = rk4_step(
                    sensitivity_state_b,
                    dt
                )


    # ========================================================
    # DRAW
    # ========================================================

    screen.fill(
        BLACK
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    draw_text(
        "DOUBLE PENDULUM — AI CHAOS LAB",
        25,
        20,
        WHITE,
        FONT_TITLE
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    draw_text(
        "BLUE = PHYSICAL SIMULATION",
        25,
        70,
        BLUE,
        FONT_MEDIUM
    )

    draw_text(
        "RED = LSTM RECURSIVE PREDICTION",
        25,
        98,
        RED,
        FONT_MEDIUM
    )

    draw_text(
        "GREEN / ORANGE = SENSITIVITY",
        25,
        126,
        ORANGE,
        FONT_MEDIUM
    )

    # --------------------------------------------------------
    # Actual pendulum
    # --------------------------------------------------------

    draw_pendulum(
        actual_state,
        BLUE,
        BLUE,
        actual_trail,
        trails_enabled
    )

    # --------------------------------------------------------
    # AI pendulum
    # --------------------------------------------------------

    if ai_enabled:

        draw_pendulum(
            ai_state,
            RED,
            RED,
            ai_trail,
            trails_enabled
        )

    # --------------------------------------------------------
    # Sensitivity
    # --------------------------------------------------------

    if sensitivity_enabled:

        point_a = world_to_screen(
            state_to_xy(
                sensitivity_state_a
            )
        )

        point_b = world_to_screen(
            state_to_xy(
                sensitivity_state_b
            )
        )

        pygame.draw.circle(
            screen,
            GREEN,
            point_a,
            6
        )

        pygame.draw.circle(
            screen,
            ORANGE,
            point_b,
            5
        )

    # --------------------------------------------------------
    # Horizon
    # --------------------------------------------------------

    if prediction_horizon is None:

        horizon_message = (
            "Prediction horizon: "
            "not reached"
        )

        horizon_color = GREEN

    else:

        horizon_message = (
            "Prediction horizon: "
            f"{prediction_horizon:.5f} s"
        )

        horizon_color = YELLOW

    draw_text(
        horizon_message,
        25,
        555,
        horizon_color,
        FONT_MEDIUM
    )

    # --------------------------------------------------------
    # Error graph
    # --------------------------------------------------------

    draw_error_graph()

    # --------------------------------------------------------
    # Panel
    # --------------------------------------------------------

    draw_panel()

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    pygame.display.flip()

    clock.tick(
        FPS
    )


# ============================================================
# SHUTDOWN
# ============================================================

pygame.quit()


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 75)
print("AI CHAOS LAB FINISHED")
print("=" * 75)

print(
    f"Final simulation time: "
    f"{elapsed_time:.6f} s"
)

print(
    f"Final AI x/y error: "
    f"{last_error:.8e} m"
)

if prediction_horizon is None:

    print(
        "Prediction horizon: "
        "tolerance was not exceeded."
    )

else:

    print(
        f"Prediction horizon: "
        f"{prediction_horizon:.8f} s"
    )

final_state_distance = (
    state_distance(
        sensitivity_state_a,
        sensitivity_state_b
    )
)

final_xy_distance = (
    xy_distance(
        sensitivity_state_a,
        sensitivity_state_b
    )
)

print()
print(
    "SENSITIVITY"
)

print(
    f"Initial state separation: "
    f"{sensitivity_initial_distance:.8e}"
)

print(
    f"Final state separation: "
    f"{final_state_distance:.8e}"
)

print(
    f"Final x/y separation: "
    f"{final_xy_distance:.8e} m"
)

if sensitivity_initial_distance > 0:

    print(
        f"Amplification: "
        f"{final_state_distance / sensitivity_initial_distance:.8e}x"
    )

lambda_final = (
    finite_time_lambda(
        sensitivity_initial_distance,
        final_state_distance,
        elapsed_time
    )
)

print(
    f"Finite-time Lyapunov estimate: "
    f"{lambda_final:.8e} /s"
)

print("=" * 75)
print(f"total time spent{time.time()- initial_time_entire_code} seconds")

