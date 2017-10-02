import random
import numpy
import matplotlib.pyplot as plt

from deap import base, creator, tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

ciudades=[]

IND_SIZE=10# tamaño del individuo. 
plt.figure()
x=0
y=0

for g in range(IND_SIZE):
    x=random.randint(1,40)
    y=random.randint(1,40)
    ciudades.append(complex(x,y))
    plt.subplot(2,2,1)
    plt.plot(x,y,'bo')
    plt.annotate('Ciudad {0}'.format(g),xy=(x,y))
    plt.subplot(2,2,2)
    plt.plot(x,y,'bo')    
    plt.annotate('Ciudad {0}'.format(g),xy=(x,y))

for z in range(IND_SIZE):
    print("Ciudad", z, " -> ", ciudades[z], "\n")
    
    

toolbox = base.Toolbox()
toolbox.register("attr_float", numpy.random.permutation,IND_SIZE)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.attr_float)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    dis=0
    k=individual[0]
    for i in individual[1:]:
        dis+=abs(ciudades[i]- ciudades[k])
        k=i
    return dis,

toolbox.register("mate", tools.cxOrdered)  # Cruce en dos puntos
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def main(pop):
    
    CXPB, MUTPB, NGEN = 0.5, 0.2, 80
    
    
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

# Comprobemos la población al inicio 
pop = toolbox.population(n=50)

for k, i in enumerate(pop):    
    print("Ind:",k, "->", i, " Distancia= ",evaluate(i))
    
result = main(pop)
mejor=0
primero=0
segundo=0
print("-------------------------------")
for k, i in enumerate(result):
    if(k==0):
        mejor=i
        primero=evaluate(i)
        print("Ind:",k, "->", i, " Distancia= ",primero)
    else:
        segundo=evaluate(i)
        if(primero>segundo):
            primero=segundo
            mejor=i
        print("Ind:",k, "->", i, " Distancia= ",segundo)
    

length=len(mejor)
l=1
print(mejor)
for l in range(length):
    plt.plot((ciudades[mejor[l-1]].real,ciudades[mejor[l-1]].imag),(ciudades[mejor[l-1]].real,ciudades[mejor[l-1]].imag))
plt.show()
