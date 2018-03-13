from random import *

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
    return

def diversity_calc():
    diversityValue = 0
    # if typePopulation == 1:
    #     # METHOD: Moment of Inertia http://www.joinville.udesc.br/portal/professores/parpinelli/materiais/MomentOf_Inertia_Diversity___EA_01.pdf
    #     centroid = []
    #     for i in range (chromoPopulation):
    #         aux = 0
    #         for j in range (nPopulation):
    #             aux += population[j][i]
    #         aux = aux/nPopulation
    #         centroid.append(aux)
    #     for i in range (chromoPopulation):
    #         for j in range(nPopulation):
    #             diversityValue += (population[j][i] - centroid[i])**2
    #     diversityValue = diversityValue*nPopulation
    #     return diversityValue
    if typePopulation == 2 or typePopulation == 4:
        for i in range(nPopulation-1):
            for j in range(i+1,nPopulation):
                aux = list(zip(population[i],population[j]))
                soma = 0
                for k in range(chromoPopulation):
                    soma += abs(aux[k][0] - aux[k][1])
                diversityValue += soma
        return diversityValue

    if typePopulation == 1 or typePopulation == 3:
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
    return

typePopulation = input("Escolha uma codificacao 1-BIN, 2-INT, 3-REAL e 4-INTPERM:")
typePopulation = int(typePopulation)
if typePopulation != 1 and typePopulation != 4:
    ilPopulation = input("Escolha uma limite inferior:")
    slPopulation = input("Escolha uma limite superior:")
nPopulation = input("Escolha a quantidade de individuos:")
chromoPopulation = input("Escolha o tamanho do cromossomo:")

# CAST
if typePopulation != 1 and typePopulation != 4:
    ilPopulation = int(ilPopulation)
    slPopulation = int(slPopulation)
nPopulation = int(nPopulation)
chromoPopulation = int(chromoPopulation)
# END CAST

# VARIABLES
population = []
# population = [[5,5,5,5,5],[5,5,5,5,5],[5,5,5,5,5],[5,5,5,5,5],[5,5,5,5,5],]
diversity = []
fitness = []
# END VARIABLES

#  MAIN LOOP
populate()
print(population)
diversity.append(diversity_calc())
diversity_standarlization()
# print(diversity)
fitness_calc()
print(fitness)
# END MAIN LOOP
