import numpy as np
np.random.seed(1)

# Rosenbrock関数
def rosenbrock(x, y):
    a = 1
    b = 100
    return (a - x)**2 + b * (y - x**2)**2

# n次元対応rosenbrock関数
def rosenbrock_nd(x):
    return sum((1-x[i])**2 + 100*(x[i+1] - x[i]**2)**2 for i in range(len(x)-1))



# 初期個体群の生成
def initialize_population(size, dim = 2): 
    return np.random.uniform(-1, 1, (size, dim)) #-1~1の範囲の値を持つdim次元の個体をsize個生成

# rosenblock関数の計算
def fitness(population):
    return np.array([rosenbrock_nd(ind) for ind in population])

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



# 交叉
# 2つの固体から新たな固体を生成
# 一点交叉
def crossover(parent1, parent2):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1)) #ランダムな点を選択
        return np.concatenate((parent1[:point], parent2[point:])) #ランダム点で2つを結合
    else:
        return parent1

# 二点交叉
def two_point_crossover(parent1, parent2):
    if np.random.rand() < crossover_rate:
        point1, point2 = np.sort(np.random.randint(1, len(parent1), 2))
        child = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
        return child
    else:
        return parent1

# 一様交叉
def uniform_crossover(parent1, parent2):
    if np.random.rand() < crossover_rate:
        child = np.empty_like(parent1)
        for i in range(len(parent1)): # 各遺伝子について0.5の確率で別の親から持ってくる
            if np.random.rand() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child
    else:
        return parent1

# 突然変異
def mutate(individual):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.normal(0, 1)
    return individual

# 遺伝的アルゴリズム
def genetic_algorithm(elite=False):
    population = initialize_population(population_size, 50)
    for generation in range(generations):
        fitness_values = fitness(population)

        population = selection(population, fitness_values)
        new_population = []
        for i in range(0, len(population), 2):
            parent1, parent2 = population[i], population[i+1]
            #offspring1 = uniform_crossover(parent1, parent2)
            offspring1 = crossover(parent1, parent2)
            #offspring1 = two_point_crossover(parent1, parent2)
            #offspring2 = uniform_crossover(parent2, parent1)
            offspring2 = crossover(parent2, parent1)
            #offspring2 = two_point_crossover(parent2, parent1)
            new_population.append(mutate(offspring1))
            new_population.append(mutate(offspring2))

        new_population = np.array(new_population)
        if elite:
            elite_indices = np.argsort(fitness_values)[:elite_size]
            elites = population[elite_indices]
            remove_indices = np.random.choice(len(new_population), elite_size, replace=False)
            new_population = np.delete(new_population, remove_indices, axis=0)
            new_population = np.concatenate((new_population, elites))
        
        population = new_population
        best_fitness = np.min(fitness_values)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")
    best_individual = population[np.argmin(fitness(population))]
    return best_individual



# 遺伝的アルゴリズムのパラメータ
population_size = 500  # 個体数
generations = 2000     # 世代数
mutation_rate = 0.01   # 突然変異率
crossover_rate = 0.8   # 交叉率
elite_size = 10        # エリート個体のサイズ

best_solution = genetic_algorithm(elite=False)
print(f"Best Solution: {best_solution}, Fitness: {rosenbrock_nd(best_solution)}")
