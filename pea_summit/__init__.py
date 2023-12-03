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

"""
==============================
PEA Summit (:mod:`pea_summit`)
==============================

.. currentmodule:: pea_summit

This is a small library of helper functions used during the IBM Quantum Summit
practitioners forum.

Trotter Circuit Construction
============================

.. autosummary::
    :toctree: ./stubs/

    construct_layer_couplings
    trotter_circuit
    magnetization_observables
    entangling_layer
    remove_qubit_couplings
    directed_coupling_map

Result Visualization
====================

.. autosummary::
    :toctree: ./stubs/

    plot_trotter_results
    plot_qubit_zne_data

Utilities
=========

.. autosummary::
    :toctree: ./stubs/

    extract_saved_data
    save_result
    save_json
    load_result
    load_json
"""

__version__ = "23.0.0"

from IPython import get_ipython
from .trotter_circuits import (
    trotter_circuit,
    magnetization_observables,
    entangling_layer,
    remove_qubit_couplings,
    directed_coupling_map,
    construct_layer_couplings,
)
from .visualization import plot_trotter_results, plot_qubit_zne_data
from .file_utils import (
    save_result,
    load_result,
    save_json,
    load_json,
    extract_saved_data,
)
