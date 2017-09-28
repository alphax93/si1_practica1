import random
from deap import base, creator, tools
import numpy as np
import matplotlib.pyplot as plt

# Ciudades

IND_SIZE=4
IND_SIZE=10
xmin, xmax, ymin, ymax =1, 100, 1, 100

#cities=[]
#cities.append(1+1j)
#cities.append(1+5j)
#cities.append(5+1j)
#cities.append(5+5j)
cities = [complex(random.randint(xmin,xmax),random.randint(ymin,ymax)) for i in range(IND_SIZE)]
print(cities, "\n")
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
def evaluate(individual): #Devuelve la longitud de la ruta 
    distance=0
    j = individual[0]
    for i in individual[1:]:
        distance += abs(cities[i]-cities[j])
        j = i
    return distance,  # Fíjate en la coma al final, quiere decir que devolvemos una tupla.

toolbox.register("mate", tools.cxOrdered)  
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


hof = tools.HallOfFame(1)
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("FB", np.min)
stats.register("A", np.mean)

logbook = tools.Logbook()

def main(pop):
    
    CXPB, MUTPB, NGEN = 0.5, 0.2, 100

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
        
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=g, **record)
        print("Gen", g+1 )
        print("\tBest Individual:", hof[0])
        print("\tFitness of Best:", record["FB"])
        print("\tAverage Fitness:", record["A"])
        
    return pop

pop = toolbox.population(n=20)

hof.update(pop)
print("Gen 0 - Initial Population")
print("\tBest Individual:", hof[0])
print("\tFitness of Best:", toolbox.evaluate(hof[0])[0])
   
result = main(pop)

print("-------------------------------\n")

hof.update(pop)
print("Best Individual:", hof[0], "---> Fitness =", hof[0].fitness.values[0],"\n" )


reals = np.array([int(i.real) for i in cities])
imags = np.array([int(i.imag) for i in cities])



#First subplot: Cities
plt.subplot(221)

plt.axis([xmin-1,xmax+1,ymin-1,ymax+1])
plt.plot(reals,imags,'bo')
plt.title("Cities")
plt.xlabel("X Coord")
plt.ylabel("Y Coord")
labels = ['City {0}'.format(i) for i in range(IND_SIZE)]

for label, x , y in zip(labels,reals,imags):
    plt.annotate(label, xy=(x,y),xytext=(5,5), textcoords='offset points',
                 ha='right')

#Second subplot: Best route
plt.subplot(222)

hof=np.array(hof)
r=hof[0][0]

plt.axis([xmin-1,xmax+1,ymin-1,ymax+1])
plt.plot(reals,imags,'bo')
plt.title("Best Route")
plt.xlabel("X Coord")
plt.ylabel("Y Coord")
labels = ['City {0}'.format(i) for i in range(IND_SIZE)]

for label, x , y in zip(labels,reals,imags):
    plt.annotate(label, xy=(x,y),xytext=(5,5), textcoords='offset points',
                 ha='right')

for x in hof[0][1:]:
    plt.annotate("",xy=(reals[r],imags[r]),xycoords='data',xytext=(reals[x],imags[x]), textcoords='data',
                 arrowprops=dict(arrowstyle="<-",color="0.5"))
    r=x

#Third subplot: Average Fitness
plt.subplot(223)

gen, minFit, AvFit = logbook.select("gen","FB","A")
plt.plot(gen,AvFit,'b-')
plt.title("Average Fitness")
plt.xlabel("Generation")
plt.ylabel("Fitness")

#Fourth subplot: Fitness of Best
plt.subplot(224)

plt.plot(gen,minFit,'b-')
plt.title("Fitness of Best Individual")
plt.xlabel("Generation")
plt.ylabel("Fitness")

plt.show()
