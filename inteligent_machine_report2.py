import numpy as np

# Rosenbrock関数
def rosenbrock(x, y):
    a = 1
    b = 100
    return (a - x)**2 + b * (y - x**2)**2

# 遺伝的アルゴリズムのパラメータ
population_size = 100  # 個体群のサイズ
generations = 1000     # 世代数
mutation_rate = 0.01   # 突然変異率
crossover_rate = 0.7   # 交叉率

# 初期個体群の生成
def initialize_population(size):
    return np.random.uniform(-5, 5, (size, 2))

# rosenblock関数の計算
def fitness(population):
    return np.array([rosenbrock(ind[0], ind[1]) for ind in population])

# 次世代の親固体の選択
# トーナメント選択
def selection(population, fitness_values):
    selected = []
    for _ in range(len(population)):
        i, j = np.random.randint(0, len(population), 2) #ランダムに2つの固体を選択
        if fitness_values[i] < fitness_values[j]: #良い個体を親固体として選択しリストに追加
            selected.append(population[i])
        else:
            selected.append(population[j])
    return np.array(selected) #次世代の親固体リスト

# 交叉 (一点交叉)
# 2つの固体から新たな固体を生成
def crossover(parent1, parent2):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1)) #ランダムな点を選択
        return np.concatenate((parent1[:point], parent2[point:])) #ランダム点で2つを結合
    else:
        return parent1

# 突然変異
def mutate(individual):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.normal(0, 1)
    return individual

# 遺伝的アルゴリズムの実行
def genetic_algorithm():
    population = initialize_population(population_size)
    for generation in range(generations):
        fitness_values = fitness(population)
        population = selection(population, fitness_values)
        new_population = []
        for i in range(0, len(population), 2):
            parent1, parent2 = population[i], population[i+1]
            offspring1 = crossover(parent1, parent2)
            offspring2 = crossover(parent2, parent1)
            new_population.append(mutate(offspring1))
            new_population.append(mutate(offspring2))
        population = np.array(new_population)
        best_fitness = np.min(fitness_values)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")
    best_individual = population[np.argmin(fitness(population))]
    return best_individual

best_solution = genetic_algorithm()
print(f"Best Solution: {best_solution}, Fitness: {rosenbrock(best_solution[0], best_solution[1])}")


