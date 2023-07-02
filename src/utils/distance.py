import torch
import numpy as np
from sklearn.metrics import jaccard_score

def jaccard_neigh(pred, attr):

    def jaccard_sim(vec):
        sim_list = [jaccard_score(vec, attr[i, :]) for i in range(attr.shape[0])]
        return np.array(sim_list)

    sim_mat = np.array([list(jaccard_sim(pred[i, :])) for i in range(pred.shape[0])])
    neighs = np.argmax(sim_mat, axis=1)
    return neighs