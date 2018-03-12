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
    return

def diversity(diversityValue):
    if typePopulation == 1:
        # METHOD: Moment of Inertia http://www.joinville.udesc.br/portal/professores/parpinelli/materiais/MomentOf_Inertia_Diversity___EA_01.pdf
        centroid = []
        for i in range (cromoPopulation):
            aux = 0
            for j in range (nPopulation):
                aux += population[j][i]
            aux = aux/nPopulation
            centroid.append(aux)
        for i in range (cromoPopulation):
            for j in range(nPopulation):
                diversityValue += (population[j][i] - centroid[i])**2
        diversityValue = diversityValue*nPopulation
        return diversityValue

    if typePopulation == 2 or typePopulation == 4:
        for i in range(nPopulation-1):
            for j in range(i+1,nPopulation):
                aux = zip(population[i],population[j])
                soma = 0
                for k in range(cromoPopulation):
                    soma += abs(aux[k][0] - aux[k][1])
                diversityValue += soma
        return diversityValue

    if typePopulation == 3:
        # METHOD: Moment of Inertia http://www.joinville.udesc.br/portal/professores/parpinelli/materiais/MomentOf_Inertia_Diversity___EA_01.pdf
        centroid = []
        for i in range (cromoPopulation):
            aux = 0
            for j in range(nPopulation):
                aux += population[j][i]
            aux = aux/nPopulation
            centroid.append(aux)
        for i in range (cromoPopulation):
            for j in range(nPopulation):
                 diversityValue += (population[j][i] - centroid[i])**2
        return diversityValue

typePopulation = input("Escolha uma codificacao 1-BIN, 2-INT, 3-REAL e 4-INTPERM:")
if typePopulation != 1 and typePopulation != 4:
    ilPopulation = input("Escolha uma limite inferior:")
    slPopulation = input("Escolha uma limite superior:")
nPopulation = input("Escolha a quantidade de individuos:")
cromoPopulation = input("Escolha o tamanho do cromossomo:")

population = []
diversityValue = 0

#  MAIN LOOP
populate()
print(population)
diversityValue=diversity(diversityValue)
print(diversityValue)
# END MAIN LOOP
