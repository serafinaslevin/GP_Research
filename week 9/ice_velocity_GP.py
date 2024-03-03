


from deap import base, creator, tools, gp, algorithms
import numpy as np
import pandas as pd
import operator
import random
from sklearn.metrics import mean_squared_error
import math 
from math import sqrt, sin, cos, log
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('data/trimmed_ice_velocity.csv')

# Split data into features and target
X = data[["x-axis", "y-axis", "precipitation", "air_temp", "ocean_temp"]]
y = data["ice_velocity"]

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert training data to format expected by the evalFitness function
training_points = [(row[:5], row[5]) for row in X_train.join(y_train).itertuples(index=False)]


#we can set up the protected division and log functions outside of the method as they do not change per run 
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def protectedLog(x):
    if x > 0:
        return log(x)
    else:
        return 0
    

def set_up_deap_environment(training_points, seed):
    
    #fields/parameters here 
    max_tree_depth = 5
    ERC_min = -1 #lower bound for ephemeral random constants
    ERC_max = 1 
    initial_pop_max_depth = 3 #upper bound for depth of individuals in the initial population 
    
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    
    #   Defining the terminal set 
    primitives = gp.PrimitiveSet("PrimitiveSet", arity=5)  # arity = max number of unique input arguments for an individual -> unrealted to number of functions, only number of terminals 
    primitives.renameArguments(ARG0='x_axis', ARG1='y_axis', ARG2='precipitation', ARG3='air_temp', ARG4='ocean_temp')
    
    # to change rqange of ERC, change range from -1,1 to whatever you want below vv
    primitives.addEphemeralConstant(f"EphemeralRand_{seed}", lambda: random.randint(ERC_min,ERC_max))  #means you can have a terminal constant as part of the tree

    #Defining the function set
    primitives.addPrimitive(operator.add, 2)
    primitives.addPrimitive(operator.sub, 2)
    primitives.addPrimitive(operator.mul, 2)
    primitives.addPrimitive(protectedDiv, 2) 
    primitives.addPrimitive(operator.neg, 1) #one for negation, to do operations like x-5 or -1
    primitives.addPrimitive(protectedLog, 1) 
    primitives.addPrimitive(operator.abs, 1)
    primitives.addPrimitive(math.tan, 1)
    primitives.addPrimitive(math.sin, 1)
    primitives.addPrimitive(math.cos, 1)


    # Define the toolbox
    toolbox = base.Toolbox() #initialise the toolbox, which will contain all the functions and classes we need to create our GP
    
    #generates initial pop, with min depth 1 and max intial depth 3
    #uses the genHalfAndHalf function from the deap library, which generates a tree with either full or grow method (ramped half n half)
    toolbox.register("generateInitialPop", gp.genHalfAndHalf, pset=primitives, min_=1, max_=initial_pop_max_depth) 

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
    def evalFitness(individual, points): #points is the rows in the dataset (each row is a point). the tree (individual) is automatically passed in
        func = toolbox.compile(expr=individual)     # Convert the tree expression in a callable function
        mse = ((func(*point) - target) ** 2 for point, target in points) #mse between the function and the target values over the given points
        return (math.sqrt(sum(mse) / len(points)), ) #returns the rmse, as a tuple, as the fitness function must return a tuple for DEAP (since it is designed to handle multiple objectives)

    # row[:5] extracts the first five elements - x and y axis, precip, ocean temp, air temp and row[5] extracts the 6th element (counting from 0)
    toolbox.register("evaluate", evalFitness, points=training_points)

    #selection, using k tournament selection with k=2
    toolbox.register("select", tools.selTournament, tournsize=10)

    #mate using one point crossover (hence the cxOnePoint)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_depth))

    #generates a tree to be used in the mutation (using the full method), with min depth 0 and max depth 2
    toolbox.register("mutationTree", gp.genFull, min_=0, max_=2)
    
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.mutationTree, pset=primitives)
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree_depth))
    

    return toolbox


# the below method I found on github, it is not my own work.
# from user ai4java, here is the link to the repo: https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python/blob/master/Chapter05/elitism.py
def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
    halloffame is used to implement an elitism mechanism. The individuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the best back to population:
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook



def run_gp(training_points, seed, pop_size, n_gens, crossover, mutation, n_elites):
    toolbox = set_up_deap_environment(training_points, seed)

    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(n_elites) #the eaSimpleWithElitism method will use the 5 in the hof as the elite  (specify no. of elites here)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    #this line runs the GP, using the population, toolbox, crossover probability, mutation probability, 
    #number of generations, and the stats and hof we defined above
    eaSimpleWithElitism(population, toolbox, crossover, mutation, n_gens, stats=stats, halloffame = hof)
    best_ind = hof[0]
    return best_ind.fitness.values[0], best_ind, toolbox.compile(expr=best_ind)

    
n_runs = 30
fitness_values = []  # Define the "fitness_values" list
best_individuals = []

best_fitness = math.inf #because we are minimising fitness, the rmse
best_individual = None
best_individual_func = None #this will be the best individual, parsed to a function

pop_size = 512
n_gens = 50
crossover = 0.8 # a number between 0 and 1. What percentage of the new population will be created using crossover -> NOT the probability of crossover 
mutation = 0.2
n_elites = 10

for i in range(n_runs): 
    seed = 101*i
    fitness, best_ind, best_ind_func = run_gp(training_points, seed, pop_size, n_gens, crossover, mutation, n_elites)
    fitness_values.append(fitness)
    best_individuals.append(best_ind)
    if fitness < best_fitness: #because we are minimising fitness, the rmse 
        best_fitness = fitness
        best_individual = best_ind
        best_individual_func = best_ind_func
    
    
average_fitness = np.mean(fitness_values)
print(f'Average Fitness over {n_runs} runs: {average_fitness}')
std_fitness = np.std(fitness_values)
print(f'Standard Deviation of Fitness over {n_runs} runs: {std_fitness}')
print(f'Best Fitness: {best_fitness}')
print(f'Best Individual: {best_individual}')

def eval_test_set(best_individual_func, X_test, y_test):
    func = best_individual_func
    mse = ((func(*point) - target) ** 2 for point, target in [(row[:5], row[5]) for row in X_test.join(y_test).itertuples(index=False)]) #mse between the function and the target values over the given points
    return math.sqrt(sum(mse) / len(X_test))

rmse_test = eval_test_set(best_individual_func, X_test, y_test)
print(f'RMSE on test set (using best individual): {rmse_test}')
 
import pygraphviz as pgv


nodes, edges, labels = gp.graph(best_individual)

g = pgv.AGraph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
g.layout(prog="dot")

for i in nodes:
    n = g.get_node(i)
    n.attr["label"] = labels[i]
g.draw("ice_velocity_tree.pdf")
 