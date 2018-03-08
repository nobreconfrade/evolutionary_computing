from random import *

def populate():
    for i in range(nPopulation):
        individual = []
        if typePopulation == 1:
            for j in range(cromoPopulation):
                individual.append(randint(0,1))
            population.append(individual)
        if typePopulation == 2:
            for j in range(cromoPopulation):
                individual.append(randint(ilPopulation,slPopulation))
            population.append(individual)
        if typePopulation == 3:
            for j in range(cromoPopulation):
                individual.append(uniform(ilPopulation,slPopulation))
            population.append(individual)
        if typePopulation == 4:
            population.append(sample(range(0,cromoPopulation),cromoPopulation))

def diversity():
    if typePopulation == 3:
        for i in range (0..cromoPopulation):
            for j in range(0..nPopulation):
                population[]

typePopulation = input("Escolha uma codificacao 1-BIN, 2-INT, 3-REAL e 4-INTPERM:")
if typePopulation != 1 and typePopulation != 4:
    ilPopulation = input("Escolha uma limite inferior:")
    slPopulation = input("Escolha uma limite superior:")
nPopulation = input("Escolha a quantidade de individuos:")
cromoPopulation = input("Escolha o tamanho do cromossomo:")

population = []

#  MAIN LOOP
populate()
diversity()
# END MAIN LOOP

print(population)
