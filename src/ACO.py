import numpy as np
import networkx as nx
import random

class AntColonyOptimizer:
    def __init__(self, maze, alpha=1.0, beta=2.0, evaporation_rate=0.5,
                 pheromone_deposit=100, num_ants=10, max_iterations=200):
        self.maze = maze
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.graph = nx.Graph()
        self.pheromones = {}
        self.start = None
        self.end = None
        self.best_path = None
        self.best_length = float('inf')
        self.rows, self.cols = maze.shape
        self._build_graph()

    def _build_graph(self):
        for i in range(self.rows):
            for j in range(self.cols):
                value = self.maze[i, j]
                if value != 1:
                    if value == 3:
                        self.start = (i, j)
                    elif value == 2:
                        self.end = (i, j)
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < self.rows and 0 <= nj < self.cols and self.maze[ni, nj] != 1:
                            self.graph.add_edge((i,j), (ni,nj), weight=1.0)
        self.pheromones = {edge: 1.0 for edge in self.graph.edges}

    def _heuristic(self, a, b):
        """Heurística basada en la distancia euclidiana al objetivo"""
        return 1.0 / (np.linalg.norm(np.array(b) - np.array(self.end)) + 1e-6)

    def _choose_next(self, current, visited):
        neighbors = []
        probs = []
        for n in self.graph.neighbors(current):
            if n not in visited:
                edge = tuple(sorted((current, n)))
                if edge in self.pheromones:
                    tau = self.pheromones[edge] ** self.alpha
                    eta = self._heuristic(current, n) ** self.beta
                    neighbors.append(n)
                    probs.append(tau * eta)
        if not neighbors:
            return None
        total = sum(probs)
        probs = [p / total for p in probs]
        return random.choices(neighbors, weights=probs, k=1)[0]

    def _generate_solution(self):
        path = [self.start]
        visited = set(path)
        current = self.start
        while current != self.end:
            
            next_node = self._choose_next(current, visited)
            if next_node is None:
                return []  # camino inválido
            path.append(next_node)
            visited.add(next_node)
            edge = tuple(sorted((current, next_node)))
            self.pheromones[edge] += 0.1  # aporte local simple
            current = next_node
        return path

    def run(self):
        for iteration in range(self.max_iterations):
            all_paths = []
            for _ in range(self.num_ants):
                path = self._generate_solution()
                if path and path[-1] == self.end:
                    all_paths.append(path)
                    if len(path) < self.best_length:
                        self.best_path = path
                        self.best_length = len(path)

            if self.best_path:
                for i in range(len(self.best_path) - 1):
                    edge = tuple(sorted((self.best_path[i], self.best_path[i + 1])))
                    self.pheromones[edge] += self.pheromone_deposit / self.best_length

            for edge in self.pheromones:
                self.pheromones[edge] *= (1 - self.evaporation_rate)

        return self.best_path

def imprimir_camino(matriz, camino):
    # Crear una copia para no modificar la original
    matriz_copia = np.array(matriz)

    for i, j in camino:
        if matriz_copia[i, j] not in (2, 3):  # no sobreescribimos inicio o fin
            matriz_copia[i, j] = 4

    for fila in matriz_copia:
        print(fila.tolist())

def imprimir_matriz_coloreada(matriz, camino):
    colores = {
        0: "\033[0m",    # blanco (camino libre)
        1: "\033[90m",   # gris (muro)
        2: "\033[92m",   # verde (comida)
        3: "\033[94m",   # azul (inicio)
        4: "\033[91m",   # rojo (camino recorrido)
    }

    # Crear copia para no modificar la matriz original
    matriz_coloreada = np.array(matriz)
    print()
    for i, j in camino:
        if matriz_coloreada[i, j] not in (2, 3):  # no sobrescribir inicio ni objetivo
            matriz_coloreada[i, j] = 4

    # Imprimir cada celda con su color correspondiente
    for fila in matriz_coloreada:
        for celda in fila:
            color = colores.get(celda, "\033[0m")
            print(f"{color}{celda}\033[0m", end=' ')
        print()
    print()


maze = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0, 0, 1, 3, 1, 0, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 2, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
])


a = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 3, 1, 0, 1],
    [0, 1, 1, 0, 1, 1, 1, 0, 1],
    [0, 1, 1, 0, 1, 1, 1, 0, 1],
    [0, 1, 1, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 2, 1, 1, 1, 0, 1, 1]
    ])

c = np.array([
    [0,0,0,0,0,0,0,0,0],
    [0,3,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,2,0,0],
    [0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0]
    ])
aco = AntColonyOptimizer(c)
mejor_camino = aco.run()

if mejor_camino:
    print("Mejor camino encontrado:", mejor_camino)
    imprimir_matriz_coloreada(c, mejor_camino)
else:
    print("No se encontró un camino válido.")
