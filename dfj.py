from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np
from collections import deque, defaultdict

## DFJ Formulierung
def extract_active_edges(solution_edges):
    subsets = defaultdict(set)
    for (i, j), value in solution_edges:
        if value > 0.5:
            subsets[i].add(j)
            subsets[j].add(i)
    return subsets

def find_subtour(start, subsets, visited):
    tour = []
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            tour.append(node)
            queue.extend(subsets[node] - visited)
    return tour

def generate_constraints_edges(subtour, edges):
    indices = [
        edges.index((i, j)) for i in subtour for j in subtour if i != j and (i, j) in edges
    ]
    return (indices, len(subtour) - 1)

def subtour_constraints_dynamic(solution_edges, n, edges):
    subsets = extract_active_edges(solution_edges)
    visited = set()
    constraints = []
    for node in range(n):
        if node not in visited:
            subtour = find_subtour(node, subsets, visited)
            if len(subtour) < n and len(subtour) > 1:
                constraint = generate_constraints_edges(subtour, edges)
                constraints.append(constraint)
    return constraints

def solve_tsp_dfj(cost_matrix):
    n = len(cost_matrix)
    edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    c = [cost_matrix[i][j] for i, j in edges]
    A_eq = np.zeros((2*n, len(edges)))
    for i in range(n):
        for j in range(n):
            if i != j:
                A_eq[i, edges.index((i, j))] = 1
                A_eq[n + i, edges.index((j, i))] = 1
    lb = np.zeros(len(edges))
    ub = np.ones(len(edges))
    bounds = Bounds(lb, ub)
    A_combined = A_eq
    b_combined = np.ones(2 * n)
    count = 0
    while True:
        res = milp(
            c=c, 
            constraints=LinearConstraint(A_combined, lb=b_combined, ub=b_combined),
            bounds=bounds,
        )
        if not res.success or count > 200:
            raise Exception("Not sucessfull")
        solution_edges = [(edges[i], res.x[i]) for i in range(len(edges)) if res.x[i] > 0.5]
        sec = subtour_constraints_dynamic(solution_edges, n, edges)
        if not sec:
            break
        new_A_sec = np.zeros((len(sec), len(edges)))
        new_b_sec = []
        for k, (indices, rhs) in enumerate(sec):
            new_A_sec[k, indices] = 1
            new_b_sec.append(rhs)
        A_combined = np.vstack([A_combined, new_A_sec])
        b_combined = np.hstack([b_combined, new_b_sec])
        count += 1
    return res, (LinearConstraint(A_combined, lb=b_combined, ub=b_combined), bounds)