import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint


def find_subtours_from_solution(x, edges, n, threshold=0.5):
    """
    Find all subtours in the current MILP solution.

    Because the degree constraints enforce exactly one outgoing and one incoming
    edge for every node, the selected edges form one or more directed cycles.
    """
    outgoing = {}

    for idx, value in enumerate(x):
        if value > threshold:
            i, j = edges[idx]
            outgoing[i] = j

    visited = set()
    subtours = []

    for start in range(n):
        if start in visited:
            continue

        tour = []
        current = start

        while current not in tour:
            tour.append(current)
            visited.add(current)

            if current not in outgoing:
                break

            current = outgoing[current]

        subtours.append(tour)

    return subtours


def build_dfj_subtour_constraint(subtour, edge_to_idx, num_edges):
    """
    Build one DFJ subtour elimination constraint.

    DFJ constraint:

        sum_{i in S, j in S, i != j} x_ij <= |S| - 1

    This prevents the nodes in S from forming an isolated subtour.
    """
    row = np.zeros(num_edges)

    for i in subtour:
        for j in subtour:
            if i != j:
                row[edge_to_idx[(i, j)]] = 1

    rhs = len(subtour) - 1

    return row, rhs


def extract_active_edges(res, edges, threshold=0.5):
    """
    Extract all selected edges from the final solution.
    """
    active_edges = []

    for idx, value in enumerate(res.x):
        if value > threshold:
            active_edges.append(edges[idx])

    return active_edges


def extract_tour(res, edges, n, start_node=0, threshold=0.5):
    """
    Extract the final Hamiltonian tour from the result.
    """
    outgoing = {}

    for idx, value in enumerate(res.x):
        if value > threshold:
            i, j = edges[idx]
            outgoing[i] = j

    tour = [start_node]
    current = start_node

    while True:
        current = outgoing[current]

        if current == start_node:
            break

        tour.append(current)

    return tour


def solve_tsp_dfj(cost_matrix, max_iterations=200, threshold=0.5):
    """
    Solve the directed Traveling Salesman Problem using the DFJ formulation
    with dynamic subtour elimination constraints.

    This is a cutting-plane implementation:
    1. Solve the assignment problem.
    2. Detect subtours.
    3. Add violated DFJ subtour constraints.
    4. Re-solve until one complete tour remains.

    Parameters
    ----------
    cost_matrix : array-like
        Square matrix containing the travel costs.
    max_iterations : int
        Maximum number of subtour elimination iterations.
    threshold : float
        Threshold for deciding whether an edge is active.

    Returns
    -------
    res : scipy.optimize.OptimizeResult
        Final MILP result.
    model_info : dict
        Additional model information useful for memory measurement,
        debugging and extracting the final tour.
    """
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    n = cost_matrix.shape[0]

    if cost_matrix.shape[0] != cost_matrix.shape[1]:
        raise ValueError("cost_matrix must be square.")

    # Decision variables x_ij for all directed edges i -> j with i != j
    edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}
    num_edges = len(edges)

    # Objective vector
    c = np.array([cost_matrix[i, j] for i, j in edges], dtype=float)

    # Binary variables
    integrality = np.ones(num_edges, dtype=int)

    # Variable bounds: 0 <= x_ij <= 1
    bounds = Bounds(
        lb=np.zeros(num_edges),
        ub=np.ones(num_edges)
    )

    # Degree constraints:
    # Every node has exactly one outgoing and one incoming edge.
    A_degree = np.zeros((2 * n, num_edges))

    for i in range(n):
        for j in range(n):
            if i != j:
                # Outgoing edge from i to j
                A_degree[i, edge_to_idx[(i, j)]] = 1

                # Incoming edge to i from j
                A_degree[n + i, edge_to_idx[(j, i)]] = 1

    degree_constraint = LinearConstraint(
        A_degree,
        lb=np.ones(2 * n),
        ub=np.ones(2 * n)
    )

    # These lists store dynamically added DFJ constraints.
    subtour_rows = []
    subtour_rhs = []

    final_constraints = [degree_constraint]

    for iteration in range(max_iterations):
        constraints = [degree_constraint]

        # Add already generated subtour elimination constraints.
        if subtour_rows:
            A_subtour = np.vstack(subtour_rows)
            b_subtour = np.array(subtour_rhs)

            subtour_constraint = LinearConstraint(
                A_subtour,
                lb=-np.inf * np.ones(len(subtour_rhs)),
                ub=b_subtour
            )

            constraints.append(subtour_constraint)

        res = milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={
                "disp": False
            }
        )

        if not res.success:
            raise RuntimeError(f"DFJ MILP failed: {res.message}")

        subtours = find_subtours_from_solution(
            x=res.x,
            edges=edges,
            n=n,
            threshold=threshold
        )

        violated_subtours = [
            tour for tour in subtours
            if 1 < len(tour) < n
        ]

        # No subtours means we found one valid Hamiltonian cycle.
        if not violated_subtours:
            final_constraints = constraints

            active_edges = extract_active_edges(
                res=res,
                edges=edges,
                threshold=threshold
            )

            tour = extract_tour(
                res=res,
                edges=edges,
                n=n,
                start_node=0,
                threshold=threshold
            )

            model_info = {
                "constraints": final_constraints,
                "bounds": bounds,
                "edges": edges,
                "active_edges": active_edges,
                "tour": tour,
                "iterations": iteration + 1,
                "num_variables": num_edges,
                "num_degree_constraints": 2 * n,
                "num_subtour_constraints": len(subtour_rows),
                "objective_value": res.fun
            }

            return res, model_info

        # Add one DFJ constraint for each detected subtour.
        for subtour in violated_subtours:
            row, rhs = build_dfj_subtour_constraint(
                subtour=subtour,
                edge_to_idx=edge_to_idx,
                num_edges=num_edges
            )

            subtour_rows.append(row)
            subtour_rhs.append(rhs)

    raise RuntimeError(
        f"DFJ did not converge after {max_iterations} iterations."
    )