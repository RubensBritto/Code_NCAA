import numpy as np
import pandas as pd
import math
import statistics
import random
import secrets
import time
from numpy.random import default_rng
import sklearn
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier

# Gloval Variables
POPULATION = []
NEW_POPULATION = []
length_population = 10
length_chromosome = 13
points = []
score_points = []
CROSSOVER_RATE = 75
MUTATION_RATE = 30
good_number = 0

newDict = {}
features = ['GP', 'Ortg', 'TPA', 'adjoe', 'rimmade/(rimmade+rimmiss)', 'dunksmade', 'dunksmiss+dunksmade', 'adrtg', 'dporpag', 'stops', 'gbpm', 'stl', 'blk', 'pts']
ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

for i, j in zip(ids, features):
    newDict[i] = j

def monstro(population):
    array = []
    arrayAux = []
    for ind in population:
        elementos_repetidos = []
        cont = 0
        for i in ind.schema:
            if len(elementos_repetidos) == 0 and cont == len(ind.schema) - 1:
                arrayAux.append(ind.schema)
            if ind.schema.count(i) > 1 and i not in elementos_repetidos:
                elementos_repetidos.append(i)
            cont+=1
    array.append(Chromosome(arrayAux))
    return array


# AG
def decisionTree(features):
    data = pd.read_csv("newDataSet.csv")
    X = data[features]
    y = data["ROUND"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    dtree = DecisionTreeClassifier(criterion= 'entropy', max_depth= 4, 
                                   max_features= None, min_samples_leaf = 1, 
                                   splitter= 'best').fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    # print(f'Acuracia: {metrics.accuracy_score(y_test, y_pred)}')
    return metrics.accuracy_score(y_test, y_pred)

# Class Chromosome

class Chromosome:
    score = 0

    def __init__(self, schema):
        self.schema = schema

    def __str__(self):
        toString = ''
        for ind in self.schema:
            toString += ind
        return toString


# Selection
def selection(population, new_population):
    merged_list = []
    merged_list.extend(population)
    merged_list.extend(new_population)

    merged_list.sort(key=lambda schema: schema.score, reverse=True)

    return merged_list[:len(POPULATION)]

# mutação por complemento
def mutation(population):
    # array = []
    for i in range(len(population)):
        # arrayAux = []
        for j in range(len(population[i].schema)):
            yes = np.random.randint(0, 100)
            if(yes <= MUTATION_RATE):
                vertice = population[i].schema[j]
                index = ids.index(vertice)
                result = (length_chromosome - 1) - index
                population[i].schema[j]= ids[result]

# CrossOver

def crossOver(population):
    while len(NEW_POPULATION) < len(POPULATION):
        father = population[np.random.randint(0, len(population))].schema
        mother = population[np.random.randint(0, len(population))].schema
        yes = np.random.randint(0, 100)

        if father != mother and yes <= CROSSOVER_RATE:
            child = []
            cut = np.random.randint(1, len(father))
            child.append(father[:cut] + mother[cut:])
            child.append(mother[:cut] + father[cut:])
            for downward in child:
                NEW_POPULATION.append(Chromosome(downward))

# Fitness - Falta configurar um algoritmo aqui
def decode(ind):
    resultado = []
    for i in ind:
        for chave, valor in newDict.items():
            if chave == i:
                resultado.append(valor)
    return resultado


def score(population_test):
    for ind in population_test:
        count = 0
        ret = decode(ind.schema)
        ind.score = decisionTree(ret)


# Population

def random():
    rng = default_rng()
    numbers = rng.choice(range(0, 14), size=length_chromosome, replace=False)
    return numbers


def init_population(length_population, length_chromosome):
    for _ in range(length_population):
        array = []
        array.extend(random())
        POPULATION.append(Chromosome(array))


# Main
list_score = []

flag = False
generation = 0
count_aux = 0
POPULATION.clear()
NEW_POPULATION.clear()
init_population(length_population, length_chromosome)


while True:
    score(POPULATION)
    crossOver(POPULATION)
    score(NEW_POPULATION)
    mutation(NEW_POPULATION)
    score(NEW_POPULATION)
    NEW_POPULATION = monstro(NEW_POPULATION)
    POPULATION = selection(POPULATION, NEW_POPULATION)
    NEW_POPULATION.clear()

    for ind in POPULATION:
        list_score.append(ind.score)

    if max(list_score) > good_number:
        good_number = max(list_score)
        count_aux += 1

    list_score.clear()

    for ind in POPULATION:
        if ind.score > 0.78 or count_aux == 5:
            flag = True

    if flag:
        print("===================================================================")
        print(
            f'Individuo: {POPULATION[0].schema} e o score dele {POPULATION[0].score} geracao {generation}')
        print("===================================================================")
        score_points.append(POPULATION[0].score)
        points.append(generation)
        break

    generation += 1
