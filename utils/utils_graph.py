import numpy as np
# from numpy.linalg import norm


def gen_adjacency_mat(e_r_dict):
    '''
    Generate adjacency matrix from e-r dictionary in conversation
    :param e_r_dict:
    :return: adjacency matrix, identity matrix
    '''
    er_vec = getER_vec(e_r_dict)
    dimension = len(er_vec)
    adjacenty_matrix = np.zeros((dimension, dimension))

    for i in range(dimension):
        if er_vec[i] in e_r_dict:  # if element of er_vector[i] is an entity
            for j in range(dimension):
                if er_vec[j] in e_r_dict[er_vec[i]]:
                    adjacenty_matrix[i][j] = 1.0
        else:  # if element of er_vector[j] is a relation
            for j in range(dimension):
                for k in range(dimension):
                    if er_vec[k] in e_r_dict and er_vec[j] in e_r_dict[er_vec[k]]:
                        adjacenty_matrix[j][k] = 1.0
    # print (adjacenty_matrix)
    return adjacenty_matrix, np.identity(adjacenty_matrix.shape[0])


def getER_vec(er_dict):
    """
    :param entities: list of entities
    :param relations: list of relations
    :return: genperate ER list -> ["brazil","coach","caps",age"]
    """
    er_vector = []
    for k, v in er_dict.items():
        er_vector.append(k)
        for r in v:
            er_vector.append(r)

    return np.array(er_vector)


def get_degree_matrix(adjacency_matrix):
    """
    :param adjacency_matrix:
    :return: return degree matrix -> example [[1. 0. 0. 0.],
                                              [0. 2. 0. 0.],
                                              [0. 0. 0. 0.],
                                              [0. 0. 0. 3.]]   diagonal numbers represents total number of connections
    """
    return np.array(np.diag(np.array(np.sum(adjacency_matrix, axis=0))))


if __name__ == '__main__':
    gen_adjacency_mat({'manchester united': ['ground',
   'chairman',
   'coach',
   'has player',
   'jersey color',
   'founded on'],
  '13 \\( first in 1930 \\)': [],
  'gaston pereiro': ['midfielder',
   'playes for',
   'position',
   'goals',
   'caps',
   'age',
   'jersey']})