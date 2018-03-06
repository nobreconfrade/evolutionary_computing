from random import *

typePopulation = input("Escolha uma codificacao 1-BIN, 2-INT, 3-FLO e 4-INTPERM:")
boundPopulation = input("Escolha uma limite:")
nPopulation = input("Escolha a quantidade de individuos:")
cromoPopulation = input("Escolha o tamanho do cromossomo:")

individual = ""
population = []

for i in range(nPopulation):
    if typePopulation == 1:
        for j in range(cromoPopulation):
            individual = randint(0,1)
            population.append(individual)
        
print(population)
