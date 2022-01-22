import numpy as np

def FullyConnectedGraph(num_agents):
    adjacency = np.ones((num_agents, num_agents))
    laplacian = -np.ones((num_agents, num_agents))
    for i in range(num_agents):
        laplacian[i,i] = num_agents
        adjacency[i,i] = 0
    return adjacency, laplacian

def RandomGraph(num_agents):
    return None