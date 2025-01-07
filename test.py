'''def parse_dimacs_geom(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    nodes = []
    for line in lines:
        if line.startswith('v'):
            _, x, y = line.split()
            nodes.append((int(x), int(y)))
    
    return nodes'''


def parse_dimacs_geom(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    nodes = []
    for line in lines:
        if line[0].isdigit():  # 检查行是否以数字开头
            x, y = line.split()
            nodes.append((int(x), int(y)))
    
    return nodes

# filepath: hopcroft_karp.py
from collections import deque

def bfs(pair_u, pair_v, dist, u_nodes, v_nodes, adj):
    queue = deque()
    for u in u_nodes:
        if pair_u[u] == 0:
            dist[u] = 0
            queue.append(u)
        else:
            dist[u] = float('inf')
    dist[0] = float('inf')
    while queue:
        u = queue.popleft()
        if dist[u] < dist[0]:
            for v in adj[u]:
                if dist[pair_v[v]] == float('inf'):
                    dist[pair_v[v]] = dist[u] + 1
                    queue.append(pair_v[v])
    return dist[0] != float('inf')

def dfs(u, pair_u, pair_v, dist, adj):
    if u != 0:
        for v in adj[u]:
            if dist[pair_v[v]] == dist[u] + 1:
                if dfs(pair_v[v], pair_u, pair_v, dist, adj):
                    pair_v[v] = u
                    pair_u[u] = v
                    return True
        dist[u] = float('inf')
        return False
    return True

def hopcroft_karp(u_nodes, v_nodes, edges):
    adj = {u: [] for u in u_nodes}
    for u, v in edges:
        adj[u].append(v)
    
    pair_u = {u: 0 for u in u_nodes}
    pair_v = {v: 0 for v in v_nodes}
    dist = {}
    
    matching = 0
    while bfs(pair_u, pair_v, dist, u_nodes, v_nodes, adj):
        for u in u_nodes:
            if pair_u[u] == 0:
                if dfs(u, pair_u, pair_v, dist, adj):
                    matching += 1
    return matching



def hungarian_algorithm(u_nodes, v_nodes, edges):
    adj = {u: [] for u in u_nodes}
    for u, v in edges:
        adj[u].append(v)
    
    pair_u = {u: 0 for u in u_nodes}
    pair_v = {v: 0 for v in v_nodes}
    dist = {}
    
    matching = 0
    while bfs(pair_u, pair_v, dist, u_nodes, v_nodes, adj):
        for u in u_nodes:
            if pair_u[u] == 0:
                if dfs(u, pair_u, pair_v, dist, adj):
                    matching += 1
    return matching

def bfs_edmonds_karp(capacity, source, sink, parent):
    visited = [False] * len(capacity)
    queue = deque([source])
    visited[source] = True
    
    while queue:
        u = queue.popleft()
        
        for v, cap in enumerate(capacity[u]):
            if not visited[v] and cap > 0:
                queue.append(v)
                visited[v] = True
                parent[v] = u
                if v == sink:
                    return True
    return False

def edmonds_karp(capacity, source, sink):
    parent = [-1] * len(capacity)
    max_flow = 0
    
    while bfs_edmonds_karp(capacity, source, sink, parent):
        path_flow = float('Inf')
        s = sink
        
        while s != source:
            path_flow = min(path_flow, capacity[parent[s]][s])
            s = parent[s]
        
        v = sink
        while v != source:
            u = parent[v]
            capacity[u][v] -= path_flow
            capacity[v][u] += path_flow
            v = parent[v]
        
        max_flow += path_flow
    
    return max_flow

def create_capacity_matrix(u_nodes, v_nodes, edges):
    size = len(u_nodes) + len(v_nodes) + 2
    capacity = [[0] * size for _ in range(size)]
    source = 0
    sink = size - 1
    
    for u in u_nodes:
        capacity[source][u] = 1
    
    for v in v_nodes:
        capacity[v][sink] = 1
    
    for u, v in edges:
        capacity[u][v] = 1
    
    return capacity, source, sink


def greedy_algorithm(u_nodes, v_nodes, edges):
    matched_u = set()
    matched_v = set()
    matching = []

    for u, v in edges:
        if u not in matched_u and v not in matched_v:
            matching.append((u, v))
            matched_u.add(u)
            matched_v.add(v)
    
    return matching.count


# filepath: main.py
import random
import json
import time
def generate_bipartite_random_edges(nodes, num_edges):
    edges = []
    node_indices = list(range(1, len(nodes) + 1))
    for _ in range(num_edges):
        u = random.choice(node_indices)
        v = random.choice(node_indices)
        if u != v:
            edges.append((u, v))
    return edges

if __name__ == "__main__":
    nodes = parse_dimacs_geom('c:/Users/123/Desktop/作业/algorithm_design/r20726.geom.txt')
    
    # Generate a random set of edges
    num_edges = 100000  # You can adjust the number of edges as needed
    #edges = generate_bipartite_random_edges(nodes, num_edges)

       # Save edges to a file
    #with open('c:/Users/123/Desktop/作业/algorithm_design/edges.json', 'w') as f:
    #    json.dump(edges, f)

    with open('c:/Users/123/Desktop/作业/algorithm_design/edges.json', 'r') as f:
        edges = json.load(f)

    
    # Create sets of nodes
    u_nodes = set(range(1, len(nodes) + 1))
    v_nodes = set(range(1, len(nodes) + 1))
    capacity, source, sink = create_capacity_matrix(u_nodes, v_nodes, edges)

    num_runs = 10
    total_time = 0
    
    for i in range(num_runs):
        start_time = time.time()
        #max_matching = hopcroft_karp(u_nodes, v_nodes, edges)
        #max_matching = hungarian_algorithm(u_nodes, v_nodes, edges)
        #max_matching = edmonds_karp(capacity, source, sink)
        max_matching = greedy_algorithm(u_nodes, v_nodes, edges)
        end_time = time.time()
        total_time += end_time - start_time

    avg_time = total_time / num_runs

    print(f"Maximum Matching: {max_matching}")
    print(f"Time taken: {avg_time} seconds")