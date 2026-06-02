import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

## MTZ Formulierung
def solve_tsp_mtz(cost_matrix):
    n = len(cost_matrix)
    edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    c = [cost_matrix[i][j] for i, j in edges] + [0] * n
    A_eq = np.zeros((2 * n, len(edges) + n))
    for i in range(n):
        for j in range(n):
            if i != j:
                edge_index = edges.index((i, j))
                A_eq[i, edge_index] = 1
                A_eq[n + i, edge_index] = 1
    u_bounds = [(1, n-1) if i > 0 else (0, 0) for i in range(n)]
    A_ineq, b_ineq = [], []
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                row = [0] * (len(edges) + n)
                edge_index = edges.index((i, j))
                row[edge_index] = n - 1
                row[len(edges) + i] = 1
                row[len(edges) + j] = -1
                A_ineq.append(row)
                b_ineq.append(n - 2)
    lb_x = np.zeros(len(edges))
    ub_x = np.ones(len(edges))
    lb_u = [bound[0] for bound in u_bounds]
    ub_u = [bound[1] for bound in u_bounds]
    lb = np.concatenate((lb_x, lb_u))
    ub = np.concatenate((ub_x, ub_u))
    bounds = Bounds(lb, ub)
    res = milp(
        c=c,
        constraints=[
            LinearConstraint(A_eq, np.ones(2 * n), np.ones(2 * n)),
            LinearConstraint(np.array(A_ineq), -np.inf * np.ones(len(b_ineq)), np.array(b_ineq)),
        ],
        bounds=bounds,
    )
    return res, (
        LinearConstraint(A_eq, np.ones(2 * n), np.ones(2 * n)),
        LinearConstraint(np.array(A_ineq), -np.inf * np.ones(len(b_ineq)), np.array(b_ineq)),
        bounds
    )