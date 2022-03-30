import numpy
import random

def evaluate(individual):
    """
    Recebe um indivíduo (lista de inteiros) e retorna o número de ataques
    entre rainhas na configuração especificada pelo indivíduo.
    Por exemplo, no individuo [2,2,4,8,1,6,3,4], o número de ataques é 10.

    :param individual:list
    :return:int numero de ataques entre rainhas no individuo recebido
    """
    attacks_count = 0

    for idx_first in range(8):
        for idx_second in range(idx_first + 1, 8):
            distance = idx_second - idx_first

            can_attack = (individual[idx_first] == individual[idx_second]
                        or individual[idx_first] == individual[idx_second] - distance
                        or individual[idx_first] == individual[idx_second] + distance)

            if can_attack:
                attacks_count += 1

    return attacks_count

def tournament(participants):
    """
    Recebe uma lista com vários indivíduos e retorna o melhor deles, com relação
    ao numero de conflitos
    :param participants:list - lista de individuos
    :return:list melhor individuo da lista recebida
    """
    best_individual = min(participants, key = evaluate)
    return best_individual

def crossover(parent1, parent2, index):
    """
    Realiza o crossover de um ponto: recebe dois indivíduos e o ponto de
    cruzamento (indice) a partir do qual os genes serão trocados. Retorna os
    dois indivíduos com o material genético trocado.
    Por exemplo, a chamada: crossover([2,4,7,4,8,5,5,2], [3,2,7,5,2,4,1,1], 3)
    deve retornar [2,4,7,5,2,4,1,1], [3,2,7,4,8,5,5,2].
    A ordem dos dois indivíduos retornados não é importante
    (o retorno [3,2,7,4,8,5,5,2], [2,4,7,5,2,4,1,1] também está correto).
    :param parent1:list
    :param parent2:list
    :param index:int
    :return:list,list
    """
    first = parent1[:index] + parent2[index:]
    second = parent2[:index] + parent1[index:]
    return first, second

def mutate(individual, m):
    """
    Recebe um indivíduo e a probabilidade de mutação (m).
    Caso random() < m, sorteia uma posição aleatória do indivíduo e
    coloca nela um número aleatório entre 1 e 8 (inclusive).
    :param individual:list
    :param m:int - probabilidade de mutacao
    :return:list - individuo apos mutacao (ou intacto, caso a prob. de mutacao nao seja satisfeita)
    """
    new_individual = individual[:]
    
    mutated = numpy.random.uniform(low = 0, high = 1) < m
    if mutated:
        position = numpy.random.randint(low = 0, high = 8)
        new_value = numpy.random.randint(low = 0, high = 8)
        new_individual[position] = new_value

    return new_individual

def run_ga(g, n, k, m, e):
    """
    Executa o algoritmo genético e retorna o indivíduo com o menor número de ataques entre rainhas
    :param g:int - numero de gerações
    :param n:int - numero de individuos
    :param k:int - numero de participantes do torneio
    :param m:float - probabilidade de mutação (entre 0 e 1, inclusive)
    :param e:bool - se vai haver elitismo
    :return:list - melhor individuo encontrado
    """

    elements = []

    individuals = numpy.random.randint(low = 1, high = 9, size = (n, 8))
    for ind in individuals:
        elements.append(list(ind))

    print("elements",elements)
    for _ in range(g):
        if e:
            new_els = [tournament(elements)]
        else:
            new_els = []
        
        while len(new_els) < n:
            k1 = random.sample(elements, k)
            k2 = random.sample(elements, k)
            els_frst = tournament(k1)
            els_scnd = tournament(k2)
            
            ext_frst, ext_scnd = crossover(els_frst, els_scnd, numpy.random.randint(low = 0, high = 8))
            ext_frst = mutate(ext_frst, m)
            ext_scnd = mutate(ext_scnd, m)

            new_els.extend([ext_frst, ext_scnd])

        elements = new_els

    return tournament(elements)

print("run", run_ga(5, 6, 6, 0.8, False))