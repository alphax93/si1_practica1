import random
from deap import base, creator, tools
import numpy as np

# Ciudades

IND_SIZE=5

cities = [complex(random.randint(1,10),random.randint(1,10)) for i in range(IND_SIZE)]
print(cities)
cities=np.array(cities)

# Fitness (-0.1 para minimizar)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Individuo
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox=base.Toolbox()
toolbox.register("perm", np.random.permutation,IND_SIZE)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.perm)

# Poblacion
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operadores
def evaluate(individual):  # Nuestra función de fitness será tan solo la suma de los 10 valores del individuo.
    distance=0
    for i in range(IND_SIZE-1):
        distance += abs(cities[individual[i]]-cities[individual[i+1]])
    return distance,  # Fíjate en la coma al final, quiere decir que devolvemos una tupla.

toolbox.register("mate", tools.cxOrdered)  
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def main(pop):
    
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    # Evaluate the entire population
    fitnesses = [toolbox.evaluate(i) for i in pop]
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = [toolbox.clone(i) for i in offspring]

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [toolbox.evaluate(i) for i in invalid_ind] 
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop = offspring

    return pop

pop = toolbox.population(n=10)

for k, i in enumerate(pop):

    print("Ind:",k, "->", i, "->", toolbox.evaluate(i))
    
result = main(pop)

print("-------------------------------")

for k, i in enumerate(result):
    
    print("Ind:",k, "->", i, "->", toolbox.evaluate(i))
