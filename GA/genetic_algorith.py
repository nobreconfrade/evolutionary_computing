import matplotlib.pyplot as plt
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
            binPopulation = bin_population_calc(ilPopulation,slPopulation,precisionPopulation)
            for j in range(binPopulation):
                individual.append(randint(0,1))
            population.append(individual)
        if typePopulation == 6:
            bin1Population = bin_population_calc(il1Population,sl1Population,precisionPopulation)
            bin2Population = bin_population_calc(il2Population,sl2Population,precisionPopulation)
            for j in range(bin1Population+bin2Population):
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

    if typePopulation == 1 or typePopulation == 3 or typePopulation == 5 or typePopulation == 6:
        # METHOD: Moment of Inertia http://www.joinville.udesc.br/portal/professores/parpinelli/materiais/MomentOf_Inertia_Diversity___EA_01.pdf
        centroid = []
        # gambiarra forte para pegar a quantidade de genes do primeiro individuo
        for i in range(len(population[0])):
            aux = 0
            for j in range(nPopulation):
                aux += population[j][i]
            aux = aux/nPopulation
            centroid.append(aux)
        for i in range(len(population[0])):
            for j in range(nPopulation):
                 diversityValue += (population[j][i] - centroid[i])**2
        return diversityValue

def diversity_standarlization(dl):
    big = max(dl)
    l = []
    for i in dl:
        l.append((i-0)/(big-0))
    return l

def fitness_calc():
    l = []
    if typePopulation == 1:
        for i in range(nPopulation):
            aux = 0
            for j in range(chromoPopulation-1):
                if population[i][j] != population[i][j+1]:
                    aux += 1
            l.append(aux)
    if typePopulation == 2:
        for i in range(nPopulation):
            aux = 0
            for j in range(chromoPopulation-1):
                if (int(population[i][j]%2)==0 and int(population[i][j+1]%2)==1) or (int(population[i][j]%2)==1 and int(population[i][j+1]%2)==0):
                    aux += 1
            l.append(aux)
    if typePopulation == 3:
        for i in range(nPopulation):
            firstSum = 0
            secondSum = 0
            for c in population[i]:
                firstSum += c**2.0
                secondSum += cos(2.0*pi*c)
            aux = -20.0*exp(-0.2*sqrt(firstSum/chromoPopulation)) - exp(secondSum/chromoPopulation) + 20 + e
            aux = fitness_standard_ackley(aux)
            l.append(aux)
    if typePopulation == 4:
        for i in range(nPopulation):
            aux = 0
            for j in range(len(population[i])):
                if j == len(population[i]) - 1:
                    c1 = population[i][j]
                    c2 = population[i][0]
                    l1 = traveling[c1]
                    l2 = traveling[c2]
                    aux += sqrt(pow(l1[0] - l2[0],2) + pow(l1[1] - l2[1],2))
                else:
                    c1 = population[i][j]
                    c2 = population[i][j+1]
                    l1 = traveling[c1]
                    l2 = traveling[c2]
                    aux += sqrt(pow(l1[0] - l2[0],2) + pow(l1[1] - l2[1],2))
            aux = fitness_standard_traveling(aux)
            l.append(aux)
        # print('POP',population)
        # print('FITNESS',l)
    if typePopulation == 5:
        # fitness da funcao algebrica para maximizar o valor
        binPopulation = bin_population_calc(ilPopulation,slPopulation,precisionPopulation)
        for i in range(nPopulation):
            x = bin_to_real(population[i],0,binPopulation)
            aux = cos(20*x) - (abs(x)/2) + ((x**3)/4)
            aux = fitness_standard_algebra(aux)
            aux = fitness_penalization(aux)
            l.append(aux)
    if typePopulation == 6:
        # fitness da funcao do problema da radio
        for i in range(nPopulation):
            st = bin_to_real_radio(population[i],0,bin1Population,0)
            lx = bin_to_real_radio(population[i],bin1Population+1,bin2Population+bin1Population,1)
            funcObj = (30*st+40*lx)/1360
            penalty = 0
            if st+2*lx > 40:
                penalty = (st+2*lx)/16
            r = -1
            aux = funcObj + r*penalty
            l.append(aux)
    return l

def bin_population_calc(ilPopulation,slPopulation,precisionPopulation):
    i = 0
    domainRange = ((slPopulation - ilPopulation) + 1)/ 10**-precisionPopulation
    while domainRange > 2**i:
        i+=1
    return i

def bin_to_real_radio(individual,lowerBit,upperBit,t):
    auxList = []
    for i in range(lowerBit,upperBit):
        auxList.append(individual[i])
    d = int(''.join(map(str,auxList)),2)
    if t == 0:
        x = il1Population + ((sl1Population - il1Population)/float((2**bin1Population) - 1)) * d
    if t == 1:
        x = il2Population + ((sl2Population - il2Population)/float((2**bin2Population) - 1)) * d
    return x

def fitness_standard_algebra(x):
    # standarlization for algebra function problem
    x = (4 + x)/(2 + 4)
    return x

def fitness_standard_traveling(x):
    # standarlization for Traveling Salesman Problem
    x = 13 - x
    return x

def fitness_standard_ackley(x):
    # standarlization for algebra function problem
    x = 23 - x
    return x

def fitness_standarlization(af,bf):
    big = max(bf)
    la = []
    lb = []
    for i in af:
        la.append((i)/(big))
    for i in bf:
        lb.append((i)/(big))
    return la,lb

def fitness_penalization(x):
    return x

def bin_to_real(individual,lowerBit,upperBit):
    auxList = []
    for i in range(lowerBit,upperBit):
        auxList.append(individual[i])
    d = int(''.join(map(str,auxList)),2)
    x = ilPopulation + ((slPopulation - ilPopulation)/float((2**bin_population_calc(ilPopulation,slPopulation,precisionPopulation)) - 1)) * d
    return x

def selection(fitness,g,tour):
    newPopulation = []
    '''ROULETTE'''
    newPopulation = roulette(newPopulation,fitness)
    # '''TOURNAMENT'''
    # if(g == 0):
    #     tour = int(input("Escolha o tamanho do torneio:\n"))
    # newPopulation = tournament(newPopulation,fitness,tour)
    return newPopulation

def roulette(newPopulation,fitness):
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
                    fitnessProb.append(val/fitnessSum)
            else:
                fitnessProb.append(val/fitnessSum)
        l = numpy.cumsum(fitnessProb)
        tip = uniform(0,1)
        for i,val in enumerate(l):
            if tip <= val:
                skip = i
                newPopulation.append(population[i].copy())
                break
    return newPopulation

def tournament(newPopulation,fitness,k):
    indFitness = -1
    for pop in range(nPopulation):
        maxFitness = 0
        if pop%2 == 1:
            flag = 1
            while flag:
                select = sample(range(0,nPopulation),k)
                if indFitness not in select:
                    # print("#######################################")
                    # print("indFitness:",indFitness,"select",select)
                    # print("#######################################")
                    flag = 0
        else:
            select = sample(range(0,nPopulation),k)
        for i in select:
            if maxFitness < fitness[i]:
                maxFitness = fitness[i]
                indFitness = i
        newPopulation.append(population[indFitness].copy())
    return newPopulation

def crossover_probability(p):
    n = uniform(0,1)
    if (n > p):
        return False
    else:
        return True

def crossover(newPopulation):
    lfinal = []
    prob = 0.8
    if typePopulation == 1 or typePopulation == 5 or typePopulation == 6:
        '''SINGLE'''
        for i in range(0,len(newPopulation),2):
            if (crossover_probability(prob) == False):
                lfinal.append(newPopulation[i])
                lfinal.append(newPopulation[i+1])
            else:
                cut = randint(1,len(newPopulation[i]))
                p1,p2 = single_cut(cut,newPopulation[i],newPopulation[i+1])
                lfinal.append(p1)
                lfinal.append(p2)
            # '''DOUBLE'''
            # '''UNIFORM'''
    if typePopulation == 2:
        '''SINGLE'''
        for i in range(0,len(newPopulation),2):
            if (crossover_probability(prob) == False):
                lfinal.append(newPopulation[i])
                lfinal.append(newPopulation[i+1])
            else:
                cut = randint(1,chromoPopulation)
                p1,p2 = single_cut(cut,newPopulation[i],newPopulation[i+1])
                lfinal.append(p1)
                lfinal.append(p2)
        # '''DOUBLE'''
        # '''UNIFORM'''
    if typePopulation == 3:
        '''UNIFORM AVERAGE'''
        for i in range(0,len(newPopulation),2):
            if (crossover_probability(prob) == False):
                lfinal.append(newPopulation[i])
                lfinal.append(newPopulation[i+1])
            else:
                mask = []
                for j in range(chromoPopulation):
                    mask.append(randint(0,1))
                p1,p2 = average_uniform_calc(mask,newPopulation[i],newPopulation[i+1])
                lfinal.append(p1)
                lfinal.append(p2)
    if typePopulation == 4:
        '''PMX'''
        for i in range(0,len(newPopulation),2):
            if (crossover_probability(prob) == False):
                lfinal.append(newPopulation[i])
                lfinal.append(newPopulation[i+1])
            else:
                 cut1,cut2 = sample(range(1,chromoPopulation),2)
                 p1,p2 = pmx(cut1,cut2,newPopulation[i],newPopulation[i+1])
                 lfinal.append(p1)
                 lfinal.append(p2)
    return lfinal

def single_cut(cut,p1,p2):
    p1init,p1end = bit_split_single(cut,p1)
    p2init,p2end = bit_split_single(cut,p2)
    p1 = p1init + p2end
    p2 = p2init + p1end
    return p1,p2

def bit_split_single(c,p):
    initial = []
    final = []
    for i in range(len(p)):
        if (i < c):
            initial.append(p[i])
        else:
            final.append(p[i])
    return initial,final

def average_uniform_calc(m,p1,p2):
    for i in range(chromoPopulation):
        mean = (p1[i] + p2[i]) / 2
        if (m[i] == 0):
            p1[i] = mean
        if (m[i] == 1):
            p2[i] = mean
    return p1,p2

def pmx(c1,c2,p1,p2):
    p1init,p1mid,p1end = bit_split_double(c1,c2,p1)
    p2init,p2mid,p2end = bit_split_double(c1,c2,p2)
    p1aux = p1mid
    p2aux = p2mid
    p1mid = p2aux
    p2mid = p1aux
    for i in p1init:
        if i in p1mid:
            j = p1mid.index(i)
            p1init[p1init.index(i)] = p1aux[j]
    for i in p1end:
        if i in p1mid:
            j = p1mid.index(i)
            p1end[p1end.index(i)] = p1aux[j]
    for i in p2init:
        if i in p2mid:
            j = p2mid.index(i)
            p2init[p2init.index(i)] = p2aux[j]
    for i in p2end:
        if i in p2mid:
            j = p2mid.index(i)
            p2end[p2end.index(i)] = p2aux[j]
    p1 = p1init + p1mid + p1end
    p2 = p2init + p2mid + p2end
    return p1,p2

def bit_split_double(c1,c2,p):
    initial = []
    mid = []
    final = []
    for i in range(chromoPopulation):
        if c1 < c2:
            if (i < c1):
                initial.append(p[i])
            elif (i >= c1) and (i < c2):
                mid.append(p[i])
            elif (i >= c2):
                final.append(p[i])
        else:
            if (i < c2):
                initial.append(p[i])
            elif (i >= c2) and (i < c1):
                mid.append(p[i])
            elif (i >= c1):
                final.append(p[i])
    return initial,mid,final

def mutation_probability(p):
    n = uniform(0,1)
    if (n > p):
        return False
    else:
        return True

def mutation(population):
    lfinal = []
    p = 0.03
    if typePopulation == 1 or typePopulation == 5 or typePopulation == 6:
        '''BIT FLIP'''
        for i in range(len(population)):
            for j in range(len(population[i])):
                if (mutation_probability(p) == True):
                    if (population[i][j] == 0):
                        population[i][j] = 1
                    else:
                        population[i][j] = 0
                else:
                    pass
    if typePopulation == 2:
        '''RANDOM VALUE'''
        for i in range(len(population)):
            for j in range(len(population[i])):
                if (mutation_probability(p) == True):
                    population[i][j] = randint(ilPopulation,slPopulation)
                else:
                    pass
    if typePopulation == 3:
        '''GAUSSIAN'''
        stdev = 0.1
        for i in range(len(population)):
            for j in range(len(population[i])):
                if (mutation_probability(p) == True):
                    population[i][j] = float(numpy.random.normal(population[i][j],stdev,1))
                else:
                    pass
    if typePopulation == 4:
        '''SWAP'''
        for i in range(len(population)):
            for j in range(len(population[i])):
                if (mutation_probability(p) == True):
                    flag = True
                    swap = population[i][j]
                    while flag:
                        m = randint(0,len(population[i])-1)
                        if m != j:
                            flag = False
                    population[i][j] = population[i][m]
                    population[i][m] = swap
                else:
                    pass
    return population

def elitism_find(f,p):
    aux = max(f)
    i = f.index(aux)
    best = p[i].copy()
    return best

def elitism_act(p,best):
    p.pop(0)
    p.insert(0,best)
    return p

def plotDiversity(problem,diversity):
    plt.title('Algoritmo Genético - Diversidade - Problema: '+problem)
    plt.xlim([1,len(diversity)])
    plt.ylim([0,1.1])
    plt.xlabel('Numero de gerações')
    plt.ylabel('Diversidade')

    plt.gca().set_color_cycle(['green'])

    plt.plot(range(1,len(diversity)+1), diversity)

    plt.legend(['Diversidade'], loc='upper right')

    plt.savefig("graphs/diversity/"+problem+".png")
    plt.close()
    return

def plotConvergence(problem,fitnessListAverage,fitnessListBest):
    plt.title('Algoritmo Genético - Problema: '+problem)
    plt.xlim([1,len(fitnessListBest)])
    plt.ylim([0,1.1])
    plt.xlabel('Numero de gerações')
    plt.ylabel('Fitness')

    plt.gca().set_color_cycle(['red', 'blue'])

    plt.plot(range(1, len(fitnessListAverage)+1), fitnessListAverage)
    plt.plot(range(1, len(fitnessListBest)+1), fitnessListBest)

    plt.legend(['Média dos fitness', 'Melhor fitness'], loc='upper left')

    plt.savefig("graphs/fitness/"+problem+".png")
    plt.close()
    return

def find_fitness(fl,np):
    bestList = []
    averageList = []
    for i in fl:
        mean = 0
        best = 0
        for j in i:
            mean += j
            if j > best:
                best = j
        mean = mean / np
        bestList.append(best)
        averageList.append(mean)
    return averageList,bestList

typePopulation = input("Escolha uma codificacao 1-BIN, 2-INT, 3-REAL, 4-INTPERM, 5-CODBIN e 6-RADIO:\n")
typePopulation = int(typePopulation)
if typePopulation == 6:
    il1Population = input("Escolha uma limite inferior primeira variavel:\n")
    sl1Population = input("Escolha uma limite superior primeira variavel:\n")
    il2Population = input("Escolha uma limite inferior segunda variavel:\n")
    sl2Population = input("Escolha uma limite superior segunda variavel:\n")
if typePopulation != 1 and typePopulation != 4 and typePopulation != 6:
    ilPopulation = input("Escolha uma limite inferior:\n")
    slPopulation = input("Escolha uma limite superior:\n")
nPopulation = input("Escolha a quantidade de individuos:\n")
chromoPopulation = input("Escolha o tamanho do cromossomo:\n")
if typePopulation == 5 or typePopulation == 6:
    precisionPopulation = input("Escolha a precisão da codificação:\n")
generationsPopulation = input("Escolha a quantidade gerações:\n")

# CAST
if typePopulation == 6:
    il1Population = int(il1Population)
    sl1Population = int(sl1Population)
    il2Population = int(il2Population)
    sl2Population = int(sl2Population)
if typePopulation != 1 and typePopulation != 4 and typePopulation != 6:
    ilPopulation = int(ilPopulation)
    slPopulation = int(slPopulation)
nPopulation = int(nPopulation)
chromoPopulation = int(chromoPopulation)
if typePopulation == 5 or typePopulation == 6:
    precisionPopulation = int(precisionPopulation)
generationsPopulation = int(generationsPopulation)
# END CAST

# VARIABLES
population = []
# population = [[5,5,5,5,5],[5,5,5,5,5],[5,5,5,5,5],[5,5,5,5,5],[5,5,5,5,5]]
traveling = [[0,0.2],[0.15, 0.80],[0.20, 0.65],[0.90, 0.30],[0.75, 0.45],[0.30, 0.75],[0.05, 0.05],[0.95, 0.95],[0.55, 0.55],[0.85, 0.25]]
diversityList = []
fitnessList = []
binPopulation = 0
tour = 0
if typePopulation == 1:
    problem = 'bitsalternados'
if typePopulation == 2:
    problem = 'parimpar'
if typePopulation == 3:
    problem = 'ackley'
if typePopulation == 4:
    problem = 'caixeiro'
if typePopulation == 5:
    problem = 'funcaoalgebrica'
if typePopulation == 6:
    problem = 'radio'
    bin1Population = bin_population_calc(il1Population,sl1Population,precisionPopulation)
    bin2Population = bin_population_calc(il2Population,sl2Population,precisionPopulation)
# END VARIABLES

#  MAIN LOOP
populate()
# print("---------------------------------\n")
for gen in range(generationsPopulation):
    diversityList.append(diversity_calc())
    fitness = fitness_calc()
    bestPopulation = elitism_find(fitness,population)
    newPopulation = selection(fitness,gen,tour)
    population = crossover(newPopulation)
    population = mutation(population)
    population = elitism_act(population,bestPopulation)
    fitnessList.append(fitness)
averageFitness,bestFitness = find_fitness(fitnessList,nPopulation)
diversityList = diversity_standarlization(diversityList)
averageFitness,bestFitness = fitness_standarlization(averageFitness,bestFitness)
plotDiversity(problem,diversityList)
plotConvergence(problem,averageFitness,bestFitness)
# END MAIN LOOP
