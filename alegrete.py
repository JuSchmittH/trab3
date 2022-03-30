import numpy as np


def compute_mse(theta_0, theta_1, data):
    """
    Calcula o erro quadratico medio
    :param theta_0: float - intercepto da reta
    :param theta_1: float -inclinacao da reta
    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :return: float - o erro quadratico medio
    """
    result = 1 * sum((theta_0 + theta_1*data[:,0] - data[:,1])**2)
    result = result / data.shape[0]
    return result

def step_gradient(theta_0, theta_1, data, alpha):
    """
    Executa uma atualização por descida do gradiente  e retorna os valores atualizados de theta_0 e theta_1.
    :param theta_0: float - intercepto da reta
    :param theta_1: float -inclinacao da reta
    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :param alpha: float - taxa de aprendizado (a.k.a. tamanho do passo)
    :return: float,float - os novos valores de theta_0 e theta_1, respectivamente
    """
    base = theta_0 + theta_1*data[:,0] - data[:,1]
    sum_dt0, sum_dt1 = sum(base), sum((base) * data[:,0])

    dt0, dt1 = 2 * sum_dt0 / data.shape[0], 2 * sum_dt1 / data.shape[0] 

    (new_t0, new_t1) = theta_0 - alpha * dt0,  theta_1 - alpha * dt1
    return (new_t0, new_t1)

def fit(data, theta_0, theta_1, alpha, num_iterations):
    """
    Para cada época/iteração, executa uma atualização por descida de
    gradiente e registra os valores atualizados de theta_0 e theta_1.
    Ao final, retorna duas listas, uma com os theta_0 e outra com os theta_1
    obtidos ao longo da execução (o último valor das listas deve
    corresponder à última época/iteração).

    :param data: np.array - matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    :param theta_0: float - intercepto da reta
    :param theta_1: float -inclinacao da reta
    :param alpha: float - taxa de aprendizado (a.k.a. tamanho do passo)
    :param num_iterations: int - numero de épocas/iterações para executar a descida de gradiente
    :return: list,list - uma lista com os theta_0 e outra com os theta_1 obtidos ao longo da execução
    """
    list_t0, list_t1 = [theta_0], [theta_1]
    
    for _ in range(num_iterations):
        theta_0, theta_1 = step_gradient(theta_0, theta_1, data, alpha)
        list_t0.append(theta_0)
        list_t1.append(theta_1)

    return list_t0, list_t1
