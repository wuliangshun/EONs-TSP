import time
from utils import build_U_matrix
from solver_btsp import solve_btsp

methods = ["brute", "GA", "LK", "BLKH", "Larusic", "my", "my_llm"]
U = build_U_matrix()
results = []

for method in methods:
    try:
        start = time.time()
        path, cost = solve_btsp(U, method=method)
        duration = time.time() - start

        results.append({
            "method": method,
            "path": path,
            "bottleneck": cost,
            "runtime": duration
        })

        print(f"[{method}] ✓  Cost: {cost:.3e} | Time: {duration:.3f}s")

    except NotImplementedError:
        print(f"[{method}] ✗ Not implemented")
        results.append({
            "method": method,
            "path": None,
            "bottleneck": None,
            "runtime": None
        })

# --- Save to file ---
with open("btsp_test_results.txt", "w") as f:
    f.write("BTSP Solver Comparison\n")
    f.write("=======================\n\n")
    for res in results:
        f.write(f"Method: {res['method']}\n")
        f.write(f"Path: {res['path']}\n")
        f.write(f"Bottleneck: {res['bottleneck']}\n")
        f.write(f"Runtime (s): {res['runtime']}\n")
        f.write("-" * 40 + "\n")

print("\n✓ Results saved to btsp_test_results.txt")
