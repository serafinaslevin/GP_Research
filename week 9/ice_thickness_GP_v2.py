#this is a copy of the file from week 8, which I will use to make the following improvements
# add the r-squared of a model as a second fitness function -> IS THIS STH I CAN DO???
# print out the r-squared for the best individual 
# split into training and test sets so I can use the best individual on the test set to get a better idea of how well it performs
# add an abs(x) function to the function set 
# parse the best individual to a function so it can be used automatically on the test set -> check if theres a method for this 
# add a method to print out the best individual in a readable format


from deap import base, creator, tools, gp, algorithms
import numpy as np
import pandas as pd
import operator
import random
from sklearn.metrics import mean_squared_error
import math 
from math import sqrt, sin, cos, log
from sklearn.model_selection import train_test_split

def run_gp(seed):

# inputs: x-axis, y-axis, precipitation, air_temp, ocean_temp
# output: ice_thickness


data = pd.read_csv('data/trimmed_ice_thickness.csv')
X = data[["x-axis", "y-axis", "precipitation", "air_temp", "ocean_temp"]]
y = data["ice_thickness"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# creating the fitness class. Note the -1.0 in weights, representing that lower fitness vals are better 
# the weights tuple is quite important, as it defines the number of objectives to be optimized and weather we want to minimise them or maximise them
# in this case, we are minimising just one objective, the root mean squared error which is calculated in evalFitness
# base.Fitness means that our fitness function is inheriting the base fitness class from the deap library
creator.create("Fitness", base.Fitness, weights=(-1.0,))

#creating the individual class. The second parameter gp.PrimitiveTree indicates that our custom individual class is 
# inheriting the properties of the gp.PrimitiveTree class from the deap library 
#the third parameter is specifying that our custom individuals will have an attribute named 'fitness'
#the fitness attribute in the individual will be an instance of the dynamically created class "Fitness" above 
creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)

#Defining the terminal set 
primitives = gp.PrimitiveSet("PrimitiveSet", arity=5)  # arity = max number of unique input arguments for an individual -> unrealted to number of functions, only number of terminals 
primitives.renameArguments(ARG0='x_axis', ARG1='y_axis', ARG2='precipitation', ARG3='air_temp', ARG4='ocean_temp')
# to change rqange of ERC, change range from -1,1 to whatever you want below vv
primitives.addEphemeralConstant("EphemeralRandomConstant", lambda: random.randint(-1,1))  #means you can have a terminal constant as part of the tree

#protected division function, to be used as a primitive 
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


#protected log function to be used as a primitive 
    
def protectedLog(x):
    try:
        return log(abs(x))
    except ZeroDivisionError:
        return 1
    
#Defining the function set
primitives.addPrimitive(operator.add, 2)
primitives.addPrimitive(operator.sub, 2)
primitives.addPrimitive(operator.mul, 2)
primitives.addPrimitive(protectedDiv, 2) 
primitives.addPrimitive(operator.neg, 1) #one for negation, to do operations like x-5 or -1
primitives.addPrimitive(protectedLog, 1) 


# Define the toolbox
toolbox = base.Toolbox() #initialise the toolbox, which will contain all the functions and classes we need to create our GP
#generates initial pop, with min depth 2 and max intial depth 5
#uses the genHalfAndHalf function from the deap library, which generates a tree with either full or grow method (ramped half n half)
toolbox.register("generateInitialPop", gp.genHalfAndHalf, pset=primitives, min_=2, max_=5) 

#this registers a method to create individuals with the toolbox, using the initIterate function to generate an individual with the generateInitialPop function
#creator.Individual lets us create an individual with the creator.Individual class we defined earlier
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.generateInitialPop)

#this registers a method to create a population of individuals, as defined in the line above 
#specifies that the population of individuals will be stored as a list 
#the initRepeat function specifies that we are populating the list, repeatedly calling the individual function we define above 
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#this registers a method to compile, allowing us to evaluate fitness - it compiles the tree into a function that can be called with the given primitives
toolbox.register("compile", gp.compile, pset=primitives)

# Define the fitness functions
def evalFitness(individual, points): #points is the rows in the dataset (each row is a point)
    func = toolbox.compile(expr=individual)     # Convert the tree expression in a callable function
    mse = ((func(*point) - target) ** 2 for point, target in points) #mse between the function and the target values over the given points
    return (math.sqrt(sum(mse) / len(points)), ) #returns the rmse, as a tuple, as the fitness function must return a tuple for DEAP (since it is designed to handle multiple objectives)

# row[:5] extracts the first five elements - x and y axis, precip, ocean temp, air temp and row[5] extracts the 6th element (counting from 0)
toolbox.register("evaluate", evalFitness, points=[(row[:5], row[5]) for row in X_train.join(y_train).itertuples(index=False)])

#selection, using k tournament selection with k=2
toolbox.register("select", tools.selTournament, tournsize=5)

#mate using one point crossover (hence the cxOnePoint)
toolbox.register("mate", gp.cxOnePoint)

#generates a tree to be used in the mutation (using the full method), with min depth 0 and max depth 2
toolbox.register("mutationTree", gp.genFull, min_=0, max_=2)

#mutate using the tree as above, with the default mutUniform, which mutates the tree with a uniform probability 
toolbox.register("mutate", gp.mutUniform, expr=toolbox.mutationTree, pset=primitives)


# +++++++++++++++++++++++++++++++++++++++++++++ RUNNING THE GP +++++++++++++++++++++++++++++++++++++++++++++
population = toolbox.population(n=50)
hof = tools.HallOfFame(1)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

#this line runs the GP, using the population, toolbox, crossover probability, mutation probability, 
#number of generations, and the stats and hof we defined above
algorithms.eaSimple(population, toolbox, 0.5, 0.2, 50, stats=stats, halloffame = hof)
best_ind = hof[0]
print('Best Individual = ', best_ind)
print('Best Fitness = ', best_ind.fitness.values)