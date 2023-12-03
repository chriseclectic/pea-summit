# This code is part of IBM Quantum Summit 2023.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Result visualization functions"""
# pylint: disable=singleton-comparison

from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from qiskit.primitives import EstimatorResult

# For plot style
seaborn.set()


def plot_trotter_results(
    result: EstimatorResult,
    angles: Sequence[float],
    plot_raw: bool = True,
    plot_all: bool = True,
    exact: np.ndarray = None,
    close: bool = True,
):
    """Plot average magnetization from ZNE result data.

    Args:
        result: an Estimator result obtained from this notebook.
        angles: The Rx angle values for the experiment.
        plot_raw: If True include the unextrapolated raw data curves in the plot.
        plot_all: If True plot all extrapolators, if False only plot the Automatic method.
        exact: Optional, the exact values to include in the plot. Should be a 2D numpy array.
            where the first column is theta angle values, and the second column average magnetization.
        close: Close the Matplotlib figure before returning.

    Returns:
        The figure.
    """
    values = np.asarray(result.values[0], dtype=float)
    metadata = result.metadata[0]
    num_qubits = values.shape[0]
    num_params = values.shape[1]

    angles = np.asarray(angles).ravel()
    if angles.shape != (num_params,):
        raise ValueError(
            f"Incorrect number of angles for input data {angles.size} != {num_params}"
        )

    # Get extrapolators from metadata
    # Note that the first entry is the automatic method so we skip it
    extraps = np.asarray(metadata["resilience"]["zne_extrapolator"], dtype=object)
    zne_methods = ["Automatic"] + list(extraps[0, 0, 1:, 0])

    # Get raw data noise factors from metadata
    noise_factors = np.asarray(metadata["resilience"]["zne_noise_factors"], dtype=float)
    raw_nfs = noise_factors[0, 0, 1, extraps[0, 0, 1] == None]
    zne_idx = np.arange(noise_factors.shape[-1])[noise_factors[0, 0, 1] == 0][0]

    # Take average magnetization of qubits and its standard error
    y_vals = np.mean(values, axis=0)
    y_errs = np.std(values, axis=0) / np.sqrt(num_qubits)
    x_vals = angles / np.pi

    fig, _ = plt.subplots(1, 1)
    # Plot different extrapolation methods
    for i, method in enumerate(zne_methods):
        fmt = "o-" if method == "Automatic" else "s-."
        alpha = 1 if method == "Automatic" else 0.5
        if plot_all or method == "Automatic":
            label = f"ZNE ({method})" if plot_all else "ZNE"
            plt.errorbar(
                x_vals,
                y_vals[:, i, zne_idx],
                y_errs[:, i, zne_idx],
                fmt=fmt,
                alpha=alpha,
                label=label,
            )

    # Plot raw data
    if plot_raw:
        for i, nf in enumerate(raw_nfs):
            idx = i - len(raw_nfs)
            plt.errorbar(
                x_vals,
                y_vals[:, 0, idx],
                y_errs[:, 0, idx],
                fmt="d:",
                alpha=0.5,
                label=f"Raw (nf={nf:.1f})",
            )

    # Plot exact data
    if exact is not None:
        x_exact, y_exact = exact.T
        x_exact = x_exact / np.pi
        plt.plot(x_exact, y_exact, "--", color="black", alpha=0.5, label="Exact")

    plt.ylim(0, 1.25)
    plt.xlabel("θ/π")
    plt.ylabel(r"$\overline{\langle Z \rangle}$")
    plt.legend()
    plt.title(f"Error Mitigated Average Magnetization for Rx(θ) [{num_qubits}-qubit]")
    if close:
        plt.close(fig)
    return fig


def plot_qubit_zne_data(
    result: EstimatorResult,
    angles: Sequence[float],
    qubit: int,
    num_cols: int | None = None,
    close: bool = True,
):
    """Plot ZNE extrapolation data for specific virtual qubit

    Args:
        result: The EstimatorResult for the PEA experiment.
        angles: The Rx theta angles used for the experiment.
        qubit: The virtual qubit index to plot.
        num_cols: The number of columns for the generated subplots.
        close: Close the Matplotlib figure before returning.

    Returns:
        The Matplotlib figure.
    """
    values = np.asarray(result.values[0], dtype=float)
    metadata = result.metadata[0]
    num_params = values.shape[1]

    angles = np.asarray(angles).ravel()
    if angles.shape != (num_params,):
        raise ValueError(
            f"Incorrect number of angles for input data {angles.size} != {num_params}"
        )

    stderrs = np.asarray(metadata["standard_error"], dtype=float)

    # Get extrapolators from metadata
    # Note that the first entry is the automatic method so we skip it
    extraps = np.asarray(metadata["resilience"]["zne_extrapolator"], dtype=object)[
        qubit
    ]
    zne_methods = ["Automatic"] + list(extraps[0, 1:, 0])

    # Get raw data noise factors from metadata
    noise_factors = np.asarray(
        metadata["resilience"]["zne_noise_factors"], dtype=float
    )[qubit]
    raw_idx = extraps[0, 1] == None
    zne_idx = extraps[0, 1] != None

    raw_x = noise_factors[0, 2, raw_idx]
    raw_y = values[qubit, 0, 0, raw_idx]

    # Make a square subplot
    num_cols = num_cols or int(np.ceil(np.sqrt(num_params)))
    num_rows = int(np.ceil(num_params / num_cols))
    fig, axes = plt.subplots(
        num_rows, num_cols, sharex=True, sharey=True, figsize=(12, 5)
    )
    fig.suptitle(f"ZNE data for virtual qubit {qubit}")

    for pidx, ax in zip(range(num_params), axes.flat):
        for i, method in enumerate(zne_methods):
            zne_x = noise_factors[pidx, i, zne_idx]
            zne_y = values[qubit, pidx, i, zne_idx]
            zne_yerr = stderrs[qubit, pidx, i, zne_idx]
            ax.errorbar(
                zne_x, zne_y, zne_yerr, fmt="s:", alpha=0.5, label=f"PEA ({method})"
            )

        raw_x = noise_factors[pidx, 1, raw_idx]
        raw_y = values[qubit, pidx, 1, raw_idx]
        raw_yerr = stderrs[qubit, pidx, 1, raw_idx]
        ax.errorbar(raw_x, raw_y, yerr=raw_yerr, fmt="o", label="Raw")

        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticks([0, 1, 1.6])
        ax.set_title(f"θ/π = {angles[pidx]/np.pi:.2f}")
        if pidx == 0:
            ax.set_ylabel(rf"$\langle Z_{{{qubit}}} \rangle$")
        if pidx == num_params - 1:
            ax.set_xlabel("Noise Factor")
            ax.legend()
    if close:
        plt.close(fig)
    return fig
