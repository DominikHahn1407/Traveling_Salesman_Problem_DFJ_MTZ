import numpy as np
import pandas as pd
import random
import sys
import json
import os
import time
from collections import deque, defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds
from IPython.display import clear_output


def create_distance_matrix(n, seed, max_value=0):
    rng = np.random.default_rng(seed)
    matrix = rng.integers(1000, 9999, size=(n, n), dtype=np.int64)
    np.fill_diagonal(matrix, 0)
    # Optional: make it symmetric for symmetric TSP
    cost_matrix = np.triu(matrix, 1)
    cost_matrix = cost_matrix + cost_matrix.T
    np.fill_diagonal(cost_matrix, 0)
    return cost_matrix


def get_total_size(obj, seen=None):
    if seen is None:
        seen = set()

    obj_id = id(obj)

    if obj_id in seen:
        return 0

    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(get_total_size(v, seen) for v in obj.values())
        size += sum(get_total_size(k, seen) for k in obj.keys())

    elif hasattr(obj, "__dict__"):
        size += get_total_size(obj.__dict__, seen)

    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_total_size(i, seen) for i in obj)

    return size


def list_element_sizes(input_list):
    sizes = {}

    for index, element in enumerate(input_list):
        size = sys.getsizeof(element)

        if isinstance(element, (list, dict, set, tuple)):
            contained_size = sum(sys.getsizeof(item) for item in element)
            size += contained_size

        sizes[index] = size

    return sizes


def save_results_to_file(results, i, direct_path=None):
    if direct_path != None:
        with open(direct_path, "w") as file:
            json.dump(results, file, indent=4)
    desktop_path = r"results"
    os.makedirs(desktop_path, exist_ok=True)
    file_path = os.path.join(desktop_path, f"results({i}).json")

    with open(file_path, "w") as file:
        json.dump(results, file, indent=4)

def read_tsplib_coordinates(file_path):
    coords = []
    reading_coords = False
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                reading_coords = True
                continue
            if line == "EOF":
                break
            if reading_coords:
                parts = line.split()
                if len(parts) >= 3:
                    x = float(parts[1])
                    y = float(parts[2])
                    coords.append((x, y))
    return np.array(coords)

def create_distance_matrix_from_coords(coords):
    n = len(coords)
    cost_matrix = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                cost_matrix[i, j] = int(round(np.sqrt(dx**2 + dy**2)))
    return cost_matrix