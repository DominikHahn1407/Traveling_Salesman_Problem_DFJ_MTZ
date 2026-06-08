import tracemalloc
import random
import time
from IPython.display import clear_output
from utils import read_tsplib_coordinates, create_distance_matrix_from_coords, get_total_size, save_results_to_file
from dfj import solve_tsp_dfj
from mtz import solve_tsp_mtz

def measure_time(solver, cost_matrix):
    start = time.perf_counter()
    result, result_size = solver(cost_matrix)
    end = time.perf_counter()

    return result, result_size, end - start

def measure_memory(solver, cost_matrix):
    tracemalloc.start()

    snapshot_before = tracemalloc.take_snapshot()
    result, result_size = solver(cost_matrix)
    snapshot_after = tracemalloc.take_snapshot()

    top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
    memory_bytes = sum(stat.size_diff for stat in top_stats)

    tracemalloc.stop()

    return result, result_size, memory_bytes

if __name__ == "__main__":
    for i in range(1, 6):
        cost_matrix = create_distance_matrix_from_coords(read_tsplib_coordinates("data/qa194.tsp"))
        print("Timing DFJ")
        res_dfj, res_dfj_size, dfj_time = measure_time(
            solve_tsp_dfj,
            cost_matrix
        )
        print("Timing MTZ")
        res_mtz, res_mtz_size, mtz_time = measure_time(
            solve_tsp_mtz,
            cost_matrix
        )
        print("Memory DFJ")
        _, _, dfj_memory = measure_memory(
            solve_tsp_dfj,
            cost_matrix
        )
        print("Memory MTZ")
        _, _, mtz_memory = measure_memory(
            solve_tsp_mtz,
            cost_matrix
        )
        results = {
                "dfj_time_ms": dfj_time * 1000,
                "mtz_time_ms": mtz_time * 1000,
                "dfj_size_bytes_1": get_total_size(res_dfj_size),
                "mtz_size_bytes_1": get_total_size(res_mtz_size),
                "dfj_size_bytes_2": dfj_memory,
                "mtz_size_bytes_2": mtz_memory,
            }
        save_results_to_file(results, 1, direct_path=f"results/qatar{i}.json")