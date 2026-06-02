from IPython.display import clear_output
import tracemalloc
import random
from utils import create_distance_matrix, get_total_size, save_results_to_file
from stopwatch import Stopwatch
from dfj import solve_tsp_dfj
from mtz import solve_tsp_mtz

def logic(j):
    results = []
    for i in range(5, 106, 10):
        clear_output(wait=True)
        print("Calculating at " + str(j) + "|" + str(i))
        seed = random.randint(1, 10000)
        # Run DFJ
        while True:
            try:
                random.seed(seed)
                cost_matrix = create_distance_matrix(i)
                tracemalloc.start()
                sw_dfj = Stopwatch.startNow()
                snapshot_before = tracemalloc.take_snapshot()
                res_dfj, res_dfj_size = solve_tsp_dfj(cost_matrix)
                snapshot_after = tracemalloc.take_snapshot()
                sw_dfj.stop()
                top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
                dfj_size_bytes_2 = sum(stat.size_diff for stat in top_stats)
                tracemalloc.stop()
                break
            except Exception as e:
                tracemalloc.stop()
                seed += 1
        tracemalloc.start()
        sw_mtz = Stopwatch.startNow()
        snapshot_before = tracemalloc.take_snapshot()
        res_mtz, res_mtz_size = solve_tsp_mtz(cost_matrix)
        snapshot_after = tracemalloc.take_snapshot()
        sw_mtz.stop()
        top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
        mtz_size_bytes_2 = sum(stat.size_diff for stat in top_stats)
        tracemalloc.stop()
        results.append({
            "i": i,
            "dfj_time_ms": round(sw_dfj.elapsed_milliseconds(), 10),
            "mtz_time_ms": round(sw_mtz.elapsed_milliseconds(), 10),
            "dfj_size_bytes_1": get_total_size(res_dfj_size),
            "mtz_size_bytes_1": get_total_size(res_mtz_size),
            "dfj_size_bytes_2": dfj_size_bytes_2,
            "mtz_size_bytes_2": mtz_size_bytes_2,
            "seed": seed
        })

        save_results_to_file(results, j)

if __name__ == "__main__":
    logic(100)
    exit(0)
    for i in range(1, 31, 1):
        logic(i)