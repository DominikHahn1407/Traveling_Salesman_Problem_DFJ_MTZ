import numpy as np
import pandas as pd
import random
import sys
import json
import os
import time
from stopwatch import Stopwatch
from collections import deque, defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds
from IPython.display import clear_output


seed = 312
np.random.seed(seed)

def create_distance_matrix(n, max_value=0):
    matrix = np.random.randint(1000, 9999, size=(n, n))

    np.fill_diagonal(matrix, max_value)

    for i in range(n):
        for j in range(i + 1, n):
            matrix[j, i] = matrix[i, j]

    return matrix


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


def save_results_to_file(results, i):
    desktop_path = r"results"
    os.makedirs(desktop_path, exist_ok=True)
    file_path = os.path.join(desktop_path, f"results({i}).json")

    with open(file_path, "w") as file:
        json.dump(results, file, indent=4)