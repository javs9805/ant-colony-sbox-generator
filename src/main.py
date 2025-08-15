import numpy as np
import random

# ============================
# Configuración del problema
# ============================
# Matriz de distancias (símétrica)
distances = np.array([
    [0, 2, 9, 10, 7],
    [2, 0, 6, 4, 3],
    [9, 6, 0, 8, 5],
    [10, 4, 8, 0, 6],
    [7, 3, 5, 6, 0]
])
n = len(distances)  # número de ciudades

# ============================
# Parámetros del algoritmo ACO
# ============================
num_ants = n
num_iterations = 100
alpha = 1    # importancia de la feromona
beta = 5     # importancia de la heurística (inversa de la distancia)
evaporation_rate = 0.5  # rho: tasa de evaporación
Q = 100  # constante para depósito de feromonas

# Inicialización de la matriz de feromonas
pheromone = np.ones((n, n))

def calcular_longitud_ruta(tour, distances):
    """Calcula la longitud total de una ruta, incluyendo el regreso a la ciudad de origen."""
    length = 0
    for i in range(len(tour) - 1):
        length += distances[tour[i], tour[i+1]]
    # Agregar el regreso a la ciudad de inicio
    length += distances[tour[-1], tour[0]]
    return length

def seleccionar_siguiente_ciudad(actual, visitadas, pheromone, distances, alpha, beta):
    """
    Selecciona la siguiente ciudad según una regla probabilística basada en
    las feromonas y la heurística.
    """
    # Calcula la probabilidad para cada ciudad no visitada
    prob_numerador = []
    for ciudad in range(n):
        if ciudad not in visitadas:
            tau = pheromone[actual, ciudad] ** alpha
            # Evitar división por cero (la diagonal es 0, pero se ignora ya que ciudad ya visitada)
            eta = (1.0 / distances[actual, ciudad]) ** beta if distances[actual, ciudad] != 0 else 0
            prob_numerador.append(tau * eta)
        else:
            prob_numerador.append(0)
    
    total = sum(prob_numerador)
    
    # Si total es cero (caso poco probable), selecciona una ciudad al azar de las no visitadas
    if total == 0:
        opciones = [ciudad for ciudad in range(n) if ciudad not in visitadas]
        return random.choice(opciones)
    
    # Normalizar para obtener una distribución de probabilidad
    probabilidades = [p / total for p in prob_numerador]
    # Seleccionar la siguiente ciudad según la distribución
    return np.random.choice(range(n), p=probabilidades)

# ============================
# Ejecución del ACO
# ============================

best_tour = None
best_length = float('inf')

for iteracion in range(num_iterations):
    # Almacena las rutas de todas las hormigas en esta iteración
    routes = []
    routes_length = []
    
    for ant in range(num_ants):
        # Selecciona una ciudad de inicio aleatoria
        start = random.randint(0, n-1)
        tour = [start]
        current = start
        
        # Construir la ruta hasta visitar todas las ciudades
        while len(tour) < n:
            next_city = seleccionar_siguiente_ciudad(current, tour, pheromone, distances, alpha, beta)
            tour.append(next_city)
            current = next_city
        
        # Completar el ciclo regresando a la ciudad de inicio (esto se tendrá en cuenta en el cálculo de la longitud)
        tour_length = calcular_longitud_ruta(tour, distances)
        routes.append(tour)
        routes_length.append(tour_length)
        
        # Actualizar la mejor ruta hallada
        if tour_length < best_length:
            best_length = tour_length
            best_tour = tour.copy()
    
    # Actualización de feromonas:
    # Primero: evaporación de todas las feromonas
    pheromone = (1 - evaporation_rate) * pheromone
    
    # Segundo: depósito de feromonas por cada hormiga
    for tour, tour_length in zip(routes, routes_length):
        deposit = Q / tour_length
        # Depositar en cada arista del recorrido (incluyendo el regreso a la ciudad inicial)
        for i in range(len(tour) - 1):
            pheromone[tour[i], tour[i+1]] += deposit
            pheromone[tour[i+1], tour[i]] += deposit  # Para mantener la simetría
        # Regreso a la ciudad de inicio:
        pheromone[tour[-1], tour[0]] += deposit
        pheromone[tour[0], tour[-1]] += deposit
    
    # Opcional: Imprimir información de la iteración
    print(f"Iteración {iteracion+1}: Mejor longitud encontrada hasta ahora = {best_length}")

print("\n=== Resultado Final ===")
print(f"Mejor recorrido: {best_tour} con longitud {best_length}")
