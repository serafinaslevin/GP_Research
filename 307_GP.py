import operator
import math
import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

#NOTE: I relied heavily on the example in the DEAP documentation in my code here.
#https://deap.readthedocs.io/en/master/examples/gp_symbreg.html

#Parameters
num_generations = 50
mating_probability = 0.5
mutating_probability = 0.1
pop_size = 100
selection_tournament_size = 5
max_tree_depth = 3
min_tree_depth = 1
random_seed_number = 19

X_values = [-2.0, -1.75, -1.50, -1.25, -1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 
            0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75]

Y_values = [37.0000, 24.1602, 15.0625, 8.9102, 5.0000, 2.7227, 1.5625, 1.0977, 1.0000, 1.0352, 
            1.0625, 1.0352, 1.0000, 1.0977, 1.5625, 2.7227, 5.0000, 8.9102, 15.0625, 24.1602]

pset = gp.PrimitiveSet("MAIN", 1)
pset.renameArguments(ARG0='x')  #this line is from an example on DEAP website, used to rename the 0th argument of the primive set to x 

#primitive set contains operators(+,-/ ect), the components used to build the trees - representation
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1
        
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1) #one for negation, to do operations like x-5 or -1
pset.addEphemeralConstant("123", lambda: random.randint(-1,1))  #means you can have a terminal constant as part of the tree


#initializing the creation of individuals in the population 
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

#toolbox contains functions needed - evaluation, parent selection, genetic operators (mutate, crossover)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_tree_depth, max_=max_tree_depth) #using half and half method (slide 18 lect 11)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, points):
    func = toolbox.compile(expr=individual)
    sqerrors = []
    pairs = zip(X_values, Y_values)
    for x, y in pairs:
        predicted = func(x)
        error = predicted-y 
        sqerrors.append(error**2)

    #return the average of the squared errors list 
    return math.fsum(sqerrors) / len(points),

toolbox.register("evaluate", evalSymbReg, points = X_values)
toolbox.register("select", tools.selTournament, tournsize=selection_tournament_size) #using tournament for selection, as per slides 
toolbox.register("mate", gp.cxOnePoint) #one point crossover, could try changing this 
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset) 


def main():
    random.seed(random_seed_number)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    pop, log = algorithms.eaSimple(pop, toolbox, mating_probability, mutating_probability, num_generations, halloffame=hof, verbose=False)

    print("Most successful individual found is:")
    print(hof[0])
    print("With fitness:")
    print(hof[0].fitness)

    return

if __name__ == "__main__":
    main()

