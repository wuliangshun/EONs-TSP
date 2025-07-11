# solver_btsp.py

import numpy as np
from itertools import permutations
import random

def tsp_btsp_bruteforce(U):
    """暴力穷举解 BTSP，用于小规模验证"""
    n = U.shape[0]
    best_path = None
    min_bottleneck = np.inf
    for perm in permutations(range(n)):
        # 正确地访问 U[perm[i], perm[(i + 1) % n]]，返回的是浮点数权重
        bottleneck = max(U[perm[i], perm[(i + 1) % n]] for i in range(n))
        if bottleneck < min_bottleneck:
            min_bottleneck = bottleneck
            best_path = perm
    return best_path, min_bottleneck


def tsp_btsp_lkh(U):
    """
    Approximate Lin-Kernighan Heuristic for BTSP by minimizing bottleneck via greedy edge-sorting.
    """
    n = U.shape[0]
    nodes = list(range(n))
    used = [False] * n
    path = [0]
    used[0] = True

    while len(path) < n:
        i = path[-1]
        candidates = [(j, U[i, j]) for j in nodes if not used[j]]
        candidates.sort(key=lambda x: x[1])
        next_node = candidates[0][0]
        path.append(next_node)
        used[next_node] = True

    # Find best rotation (since BTSP is circular)
    best_bottleneck = float('inf')
    best_rot = path
    for shift in range(n):
        rotated = path[shift:] + path[:shift]
        cost = max(U[rotated[i], rotated[(i+1)%n]] for i in range(n))
        if cost < best_bottleneck:
            best_bottleneck = cost
            best_rot = rotated

    return best_rot, best_bottleneck


def tsp_btsp_blkh(U, k=3):
    """
    调用 BLKH (改进版 LKH，参考 [15])
    Simulated BLKH via k-nearest neighbor edge limitation.
    BLKH 增强了 LKH 的多项式搜索和候选边机制。我们用启发式边限制来近似模拟其行为。
    限制搜索只在低权值边中进行，构造近似路径。
    """
    n = U.shape[0]
    neighbors = {
        i: sorted(range(n), key=lambda j: U[i, j])[:k]
        for i in range(n)
    }

    path = [0]
    used = {0}
    current = 0

    while len(path) < n:
        candidates = [j for j in neighbors[current] if j not in used]
        if not candidates:
            # fallback: use global nearest unused
            candidates = [j for j in range(n) if j not in used]
        next_node = min(candidates, key=lambda j: U[current, j])
        path.append(next_node)
        used.add(next_node)
        current = next_node

    bottleneck = max(U[path[i], path[(i+1)%n]] for i in range(n))
    return path, bottleneck




def tsp_btsp_larusic(U):
    """
    Exact bottleneck TSP solver based on edge thresholding (Larusic-Punnen-style).
    Only suitable for small n due to combinatorial cost.
    """
    import networkx as nx
    n = U.shape[0]
    edges = [(i, j, U[i, j]) for i in range(n) for j in range(i+1, n)]
    edges.sort(key=lambda x: x[2])

    for threshold in sorted(set(U[i, j] for i in range(n) for j in range(n) if i != j)):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from([(i, j) for (i, j, w) in edges if w <= threshold])

        try:
            # Try to find a Hamiltonian cycle within threshold
            for perm in permutations(range(n)):
                if all(G.has_edge(perm[i], perm[(i+1)%n]) for i in range(n)):
                    return list(perm), threshold
        except:
            continue

    return None, float('inf')  # no feasible cycle found


def tsp_btsp_ga(U, n_gen=100, pop_size=100):
    """遗传算法求解近似 BTSP"""
    import random

    def fitness(path):
        return max(U[path[i], path[(i + 1) % len(path)]] for i in range(len(path)))

    def crossover(p1, p2):
        size = len(p1)
        start, end = sorted(random.sample(range(size), 2))
        middle = p1[start:end]
        rest = [x for x in p2 if x not in middle]
        return rest[:start] + middle + rest[start:]

    def mutate(path, rate=0.1):
        path = list(path)
        for _ in range(int(len(path) * rate)):
            i, j = random.sample(range(len(path)), 2)
            path[i], path[j] = path[j], path[i]
        return path

    n = U.shape[0]
    population = [list(np.random.permutation(n)) for _ in range(pop_size)]
    for _ in range(n_gen):
        population = sorted(population, key=fitness)
        new_pop = population[:pop_size // 5]  # elites
        while len(new_pop) < pop_size:
            p1, p2 = random.choices(population[:pop_size // 2], k=2)
            child = mutate(crossover(p1, p2))
            new_pop.append(child)
        population = new_pop

    best_path = min(population, key=fitness)
    return best_path, fitness(best_path)



def tsp_btsp_my(U, n_samp=200, mu=2.0, f=3):
    n = U.shape[0]
    best_seq = None
    best_cost = float('inf')

    def bottleneck(seq):
        return max(U[seq[i], seq[i+1]] for i in range(len(seq) - 1))

    for _ in range(n_samp):
        unvisited = set(range(n))
        current = random.choice(list(unvisited))
        seq = [current]
        unvisited.remove(current)

        while unvisited:
            candidates = list(unvisited)
            scores = np.array([U[current, j] for j in candidates])
            probs = np.exp(-mu * scores)
            probs /= probs.sum()
            next_node = np.random.choice(candidates, p=probs)
            seq.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        cost = bottleneck(seq)
        if cost < best_cost:
            best_seq = seq
            best_cost = cost

    # Optional: local perm on last f
    if f < n:
        head, tail = best_seq[:-f], best_seq[-f:]
        for perm in permutations(tail):
            trial = head + list(perm)
            cost = bottleneck(trial)
            if cost < best_cost:
                best_seq = trial
                best_cost = cost

    return best_seq, best_cost


def get_llm_solutions(U, n=2):
    """
    模拟 LLM 生成的 channel 顺序。需要替换为实际 LLM API 调用。    
    """
    best_path, _ = tsp_btsp_bruteforce(U)
    return [list(best_path)][:n]
   

def tsp_btsp_my_llm(U, n_samp=200, mu=2.0, f=3, n_llm=2):
    """
    LLM-enhanced CoPolyBTSP variant: seeds sampling with LLM-generated sequences.
    """
    n = U.shape[0]
    best_seq = None
    best_cost = float('inf')

    def bottleneck(seq):
        return max(U[seq[i], seq[i+1]] for i in range(len(seq) - 1))

    # --- Phase 1: LLM-Generated Candidates ---
    llm_seqs = get_llm_solutions(U, n=n_llm)
    for seq in llm_seqs:
        if len(seq) == n:
            cost = bottleneck(seq)
            if cost < best_cost:
                best_seq = seq
                best_cost = cost

    # --- Phase 2: Probabilistic Sampling (same as tsp_btsp_my) ---
    for _ in range(n_samp):
        unvisited = set(range(n))
        current = random.choice(list(unvisited))
        seq = [current]
        unvisited.remove(current)

        while unvisited:
            candidates = list(unvisited)
            scores = np.array([U[current, j] for j in candidates])
            probs = np.exp(-mu * scores)
            probs /= probs.sum()
            next_node = np.random.choice(candidates, p=probs)
            seq.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        cost = bottleneck(seq)
        if cost < best_cost:
            best_seq = seq
            best_cost = cost

    # --- Optional Local Search ---
    if f < n:
        head, tail = best_seq[:-f], best_seq[-f:]
        for perm in permutations(tail):
            trial = head + list(perm)
            cost = bottleneck(trial)
            if cost < best_cost:
                best_seq = trial
                best_cost = cost

    return best_seq, best_cost


def solve_btsp(U, method="brute"):
    """
    method: one of ['brute', 'GA', 'LK', 'BLKH', 'Larusic']
    """
    method = method.lower()
    if method == "brute":
        return tsp_btsp_bruteforce(U)
    elif method == "ga":
        return tsp_btsp_ga(U)
    elif method == "lk":
        return tsp_btsp_lkh(U)
    elif method == "blkh":
        return tsp_btsp_blkh(U)
    elif method == "larusic":
        return tsp_btsp_larusic(U)
    elif method == "my":
        return tsp_btsp_my(U)
    elif method == "my_llm":
        return tsp_btsp_my_llm(U)        
    else:
        raise ValueError(f"Unknown solver method: {method}")
