from random import *
from math import *
from sys import *
import numpy

def populate():
    for i in range(nPopulation):
        individual = []
        if typePopulation == 1:
            for j in range(chromoPopulation):
                individual.append(randint(0,1))
            population.append(individual)
        if typePopulation == 2:
            for j in range(chromoPopulation):
                individual.append(randint(ilPopulation,slPopulation))
            population.append(individual)
        if typePopulation == 3:
            for j in range(chromoPopulation):
                individual.append(uniform(ilPopulation,slPopulation))
            population.append(individual)
        if typePopulation == 4:
            population.append(sample(range(0,chromoPopulation),chromoPopulation))
        if typePopulation == 5:
            binPopulation = bin_population_calc()
            for j in range(binPopulation):
                individual.append(randint(0,1))
            population.append(individual)
    return

def diversity_calc():
    diversityValue = 0
    if typePopulation == 2 or typePopulation == 4:
        for i in range(nPopulation-1):
            for j in range(i+1,nPopulation):
                aux = list(zip(population[i],population[j]))
                soma = 0
                for k in range(chromoPopulation):
                    soma += abs(aux[k][0] - aux[k][1])
                diversityValue += soma
        return diversityValue

    if typePopulation == 1 or typePopulation == 3 or typePopulation == 5:
        # METHOD: Moment of Inertia http://www.joinville.udesc.br/portal/professores/parpinelli/materiais/MomentOf_Inertia_Diversity___EA_01.pdf
        centroid = []
        for i in range (chromoPopulation):
            aux = 0
            for j in range(nPopulation):
                aux += population[j][i]
            aux = aux/nPopulation
            centroid.append(aux)
        for i in range (chromoPopulation):
            for j in range(nPopulation):
                 diversityValue += (population[j][i] - centroid[i])**2
        return diversityValue

def diversity_standarlization():
# put the standarlization function for diversity here
    return

def fitness_calc():
    if typePopulation == 1:
        for i in range(nPopulation):
            aux = 0
            for j in range(chromoPopulation-1):
                if population[i][j] != population[i][j+1]:
                    aux += 1
            fitness.append(aux)
    if typePopulation == 2 or typePopulation == 4:
        for i in range(nPopulation):
            aux = 0
            for j in range(chromoPopulation-1):
                if (int(population[i][j]%2)==0 and int(population[i][j+1]%2)==1) or (int(population[i][j]%2)==1 and int(population[i][j+1]%2)==0):
                    aux += 1
            fitness.append(aux)
    if typePopulation == 3:
        for i in range(nPopulation):
            firstSum = 0
            secondSum = 0
            for c in population[i]:
                firstSum += c**2.0
                secondSum += cos(2.0*pi*c)
            fitness.append(-20.0*exp(-0.2*sqrt(firstSum/chromoPopulation)) - exp(secondSum/chromoPopulation) + 20 + e)
    if typePopulation == 5:
        # fitness da funcao algebrica para maximizar o valor
        binPopulation = bin_population_calc()
        for i in range(nPopulation):
            x = bin_to_real(population[i],0,binPopulation)
            aux = cos(20*x) - (abs(x)/2) + ((x**3)/4)
            aux = fitness_standard_alg(aux)
            aux = fitness_penalization(aux)
            fitness.append(aux)
    return

def fitness_standard_alg(x):
    # standarlization for algebra function problem
    x = (4 + x)/(2 + 4)
    return x

def fitness_penalization(x):
    return x

def bin_population_calc():
    i = 0
    domainRange = ((slPopulation - ilPopulation) + 1)/ 10**-precisionPopulation
    while domainRange > 2**i:
        i+=1
    return i

def bin_to_real(individual,lowerBit,upperBit):
    auxList = []
    for i in range(lowerBit,upperBit):
        auxList.append(individual[i])
    d = int(''.join(map(str,auxList)),2)
    x = ilPopulation + ((slPopulation - ilPopulation)/float((2**bin_population_calc()) - 1)) * d
    return x

def selection(newPopulation):
    newPopulation = []
    newPopulation = roulette(newPopulation)
    # tournament()
    # print(newPopulation)
    return

def roulette(newPopulation):
    for pop in range(nPopulation):
        fitnessSum = 0
        fitnessProb = []
        for i,val in enumerate(fitness):
            if pop%2 == 1:
                if i == skip:
                    fitnessSum += 0
                else:
                    fitnessSum += val
            else:
                fitnessSum += val
        for i,val in enumerate(fitness):
            if pop%2 == 1:
                if i == skip:
                    fitnessProb.append(0)
                else:
                    fitnessProb.append(i/fitnessSum)
            else:
                fitnessProb.append(i/fitnessSum)
        l = numpy.cumsum(fitnessProb)
        tip = uniform(0,l[len(l)-1])
        for i,val in enumerate(l):
            if tip <= val:
                skip = i
                newPopulation.append(population[i])
                break
    return newPopulation

def tournament():

    return

typePopulation = input("Escolha uma codificacao 1-BIN, 2-INT, 3-REAL, 4-INTPERM e 5-CODBIN:\n")
typePopulation = int(typePopulation)
if typePopulation != 1 and typePopulation != 4:
    ilPopulation = input("Escolha uma limite inferior:\n")
    slPopulation = input("Escolha uma limite superior:\n")
nPopulation = input("Escolha a quantidade de individuos:\n")
chromoPopulation = input("Escolha o tamanho do cromossomo:\n")
if typePopulation == 5:
    precisionPopulation = input("Escolha a precisão da codificação:\n")

# CAST
if typePopulation != 1 and typePopulation != 4:
    ilPopulation = int(ilPopulation)
    slPopulation = int(slPopulation)
nPopulation = int(nPopulation)
chromoPopulation = int(chromoPopulation)
if typePopulation == 5:
    precisionPopulation = int(precisionPopulation)
# END CAST

# VARIABLES
population = []
# population = [[5,5,5,5,5],[5,5,5,5,5],[5,5,5,5,5],[5,5,5,5,5],[5,5,5,5,5],]
diversity = []
fitness = []
newPopulation = []
binPopulation = 0
# END VARIABLES

#  MAIN LOOP
populate()
print(population)
print("---------------------------------\n")
diversity.append(diversity_calc())
diversity_standarlization()
# print(diversity)
fitness_calc()
print(fitness)
selection(newPopulation)

# END MAIN LOOP
