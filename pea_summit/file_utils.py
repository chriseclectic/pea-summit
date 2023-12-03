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

"""Saving and loading utility functions"""

from __future__ import annotations
import json
import zipfile
import os
from pkg_resources import resource_filename

from qiskit_ibm_runtime import RuntimeDecoder, RuntimeEncoder
from qiskit.primitives import EstimatorResult


def save_json(obj: any, filename: str):
    """Save Python object to a JSON file"""
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(obj, file, cls=RuntimeEncoder, indent=4)
    print(f"JSON saved to: {filename}")


def load_json(filename: str) -> any:
    """Load JSON from a file"""
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file, cls=RuntimeDecoder)
    return data


def save_result(result: EstimatorResult, filename: str):
    """Save an Estimator result to a JSON file"""
    save_json(result.__dict__, filename)


def load_result(filename: str) -> EstimatorResult:
    """Load a saved EstimatorResult from a JSON file"""
    data = load_json(filename)
    result = EstimatorResult(**data)
    return result


def extract_saved_data():
    """Extract saved experiment data to a local `saved_data` folder."""

    # Get local path
    extract_path = os.path.join(os.getcwd())

    # Get the path to the zip file inside the package
    zip_path = resource_filename("pea_summit", "saved_data.zip")

    # Open the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Extract saved data
        for file in zip_ref.namelist():
            file_path = os.path.join(extract_path, file)
            if os.path.isdir(file_path):
                # Make a `saved_data` directory if it doesn't already exist
                os.makedirs(extract_path, exist_ok=True)
            else:
                # Check if the file already exists in the saved_data folder
                if os.path.exists(file_path):
                    print(f"Overwriting {file} in saved_data folder.")
                else:
                    print(f"Adding {file} to saved_data folder.")

                # Extract the file (this will overwrite if it already exists)
                zip_ref.extract(file, extract_path)

    print("Extraction complete.")
