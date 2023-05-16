import networkx as nx
import causaldag as cd
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import datetime
import igraph as ig
import pandas as pd
import tensorflow as tf
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid

class MyEqLayer(tf.keras.layers.Layer):
    def __init__(self, input_shape, output_shape):
        super(MyEqLayer, self).__init__()
        self.my_input_shape = input_shape
        self.my_output_shape = output_shape
    def build(self, input_shape):
        self.w1 = self.add_weight("w1", shape=(self.my_input_shape, self.my_output_shape))
        self.w2 = self.add_weight("w2", shape=(self.my_input_shape, self.my_output_shape))
        self.w3 = self.add_weight("w3", shape=(self.my_input_shape, self.my_output_shape))
        self.w4 = self.add_weight("w4", shape=(self.my_input_shape, self.my_output_shape))
        self.b  = self.add_weight("b",  shape=(self.my_output_shape,))    
    def get_config(self):
        config = super().get_config()
        config.update({
            "my_input_shape": self.my_input_shape,
            "my_input_shape": self.my_input_shape,
        })
        return config
    def call(self, inputs):
        # @tensor X1[a,b,ch2,batch] := X[a,b,ch1,batch] * w1[ch1,ch2]
        dim = inputs.shape[-1]
        one = tf.ones((dim,dim))
        X1 = tf.einsum('niab,ij->njab', inputs, self.w1)
        X2 = tf.einsum('ab,nibc,ij->njac', one, inputs, self.w2)
        # @tensor X3[a,c,ch2,batch] := X[a,b,ch1,batch] * one[b,c] * w3[ch1,ch2]
        X3 = tf.einsum('niab,bc,ij->njac', inputs, one, self.w3)
        # @tensor X4[a,d,ch2,batch] := one[a,b] * X[b,c,ch1,batch] * one[c,d] * w4[ch1,ch2]
        # @tensor X5[a,b,ch2] := one[a,b] * w5[ch2]
        X4 = tf.einsum('ab,nibc,cd,ij->njad', one, inputs, one, self.w4)
        X5 = tf.einsum('ab,j->jab', one, self.b)
        # print(X1.shape)
        # print(X2.shape)
        # print(X3.shape)
        # print(X4.shape)
        # print(X5.shape)
        output = X1 + (X2 / dim) + (X3 / dim) + (X4 / (dim * dim)) + tf.reshape(X5, [1] + X5.shape)
        # return tf.matmul(inputs, self.kernel)
        return output

class DAG_sequence(tf.keras.utils.Sequence):
    def __init__(self, params, batch_size):
        self.ngraphs, self.vals, self.nodes, self.edges, self.graph_type, self.sem_type, self.k = params
        self.batch_size = batch_size
    def __len__(self):
        return np.math.ceil(self.ngraphs / self.batch_size)
    def __getitem__(self, idx):
        if idx == 0:
            self.x, self.y = gen_train_data(
                self.ngraphs, self.vals, self.nodes, self.edges, self.graph_type, self.sem_type, self.k
            )
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                self.batch_size]
        return batch_x, batch_y

class DAG_sequence_type_ensemble(tf.keras.utils.Sequence):
    def __init__(self, params, batch_size):
        self.ngraphs, self.vals, self.nodes, self.edges, self.graph_type, self.sem_type, self.k = params
        self.batch_size = batch_size
    def __len__(self):
        return np.math.ceil(self.ngraphs / self.batch_size)
    def __getitem__(self, idx):
        if idx == 0:
            self.x, self.y = gen_train_data_type_ensemble(
                self.ngraphs, self.vals, self.nodes, self.edges, self.graph_type, self.sem_type, self.k
            )
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                self.batch_size]
        return batch_x, batch_y
    
class DAG_sequence_rho_ensemble(tf.keras.utils.Sequence):
    def __init__(self, params, batch_size):
        self.ngraphs, self.vals, self.nodes, self.edges, self.graph_type, self.sem_type, self.k = params
        self.batch_size = batch_size
    def __len__(self):
        return np.math.ceil(self.ngraphs / self.batch_size)
    def __getitem__(self, idx):
        if idx == 0:
            self.x, self.y = gen_train_data_rho_ensemble(
                self.ngraphs, self.vals, self.nodes, self.edges, self.graph_type, self.sem_type, self.k
            )
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y
    
class DAG_sequence_k_ensemble(tf.keras.utils.Sequence):
    def __init__(self, params, batch_size):
        self.ngraphs, self.vals, self.nodes, self.edges, self.graph_type, self.sem_type, self.k = params
        self.batch_size = batch_size
    def __len__(self):
        return np.math.ceil(self.ngraphs / self.batch_size)
    def __getitem__(self, idx):
        if idx == 0:
            self.x, self.y = gen_train_data_k_ensemble(
                self.ngraphs, self.vals, self.nodes, self.edges, self.graph_type, self.sem_type, self.k
            )
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y
    
class DAG_sequence_custom_barabasi(tf.keras.utils.Sequence):
    def __init__(self, params, batch_size):
        self.ngraphs, self.vals, self.nodes, self.edges, self.graph_type, self.sem_type, self.k = params
        self.batch_size = batch_size
    def __len__(self):
        return np.math.ceil(self.ngraphs / self.batch_size)
    def __getitem__(self, idx):
        if idx == 0:
            self.x, self.y = gen_train_data_custom_barabasi(
                self.ngraphs, self.vals, self.nodes, self.edges, self.graph_type, self.sem_type, self.k
            )
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.
    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold
    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est

def mydraw(adj):
    nx.draw_circular(nx.DiGraph(adj), with_labels=True, font_weight='bold', font_color='white')

def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm

def randomised_barabasi_generator(num_nodes):
    # Create a list of nodes, initially just two nodes connected by an edge
    nodes = [0, 1]
    edges = [(0, 1)]
    connectivity = 5*np.random.random()

    # Keep adding nodes until we reach the desired number
    for i in range(2, num_nodes):
        # Calculate the degree of each node in the current graph
        node_degrees = {}
        for edge in edges:
            node_degrees[edge[0]] = node_degrees.get(edge[0], 0) + 1
            node_degrees[edge[1]] = node_degrees.get(edge[1], 0) + 1

        # Choose num_edges_per_node edges to attach to the new node, based on their current degrees
        new_edges = []

        #Randomise the edge count for each node
        num_edges_rand = np.random.randint(1,5) + int((np.random.random()+1)*connectivity)
        for j in range(num_edges_rand):
            # Choose a node to attach to based on its degree
            total_degree = sum(node_degrees.values())
            rand = random.uniform(0, total_degree)
            cumulative_degree = 0
            chosen_node = None
            for node, degree in node_degrees.items():
                cumulative_degree += degree
                if rand < cumulative_degree:
                    chosen_node = node
                    break

            # Add the new edge
            new_edge = (i, chosen_node)
            new_edges.append(new_edge)
            edges.append(new_edge)
            node_degrees[i] = node_degrees.get(i, 0) + 1
            node_degrees[chosen_node] = node_degrees.get(chosen_node, 0) + 1

    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for edge in edges:
        if edge[0] != edge[1]:
            adj_matrix[edge[0], edge[1]] = 1
    return adj_matrix

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W

def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X

def simulate_nonlinear_sem(B, n, sem_type, noise_scale=None):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """
    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)
        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif sem_type == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = B.shape[0]
    scale_vec = noise_scale if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X

def get_my_normalized_cov(X):
    cor = np.corrcoef(np.transpose(X))
    vars = np.var(np.transpose(X), axis=1)
    return np.stack((cor, np.diag((vars - np.mean(vars)) / np.std(vars))))

def gen_train_data(ngraphs, vals, nodes, edges, graph_type, sem_type, k):
    # generate graphs
    # for each graph, generate data
    res_X = []
    res_Y = []
    W_ranges = ((-0.5-k, -0.5), (0.5, k+0.5))
    for _ in range(ngraphs):
        g = simulate_dag(nodes, edges, graph_type)
        # ===== Linear
        W_true = simulate_parameter(g, W_ranges)
        X = simulate_linear_sem(W_true, vals, sem_type)
        # ===== NonLinear
        #X = simulate_nonlinear_sem(g, 1000, 'mim')
        # cov = np.cov(np.transpose(X))
        cov = get_my_normalized_cov(X)
        res_X.append(cov)
        # res_Y.append((W_true != 0).astype('float32'))
        res_Y.append(g.astype('float32'))
    return np.array(res_X), np.array(res_Y)

def gen_train_data_rho_ensemble(ngraphs, vals, nodes, edges, graph_type, sem_type, k):
    # generate graphs
    # for each graph, generate data
    # edge count drawn from uniform(nodes, nodes*6)
    res_X = []
    res_Y = []
    W_ranges = ((-0.5-k, -0.5), (0.5, k+0.5))
    for _ in range(ngraphs):
        g = simulate_dag(nodes, int(np.random.uniform(nodes, nodes*6)), graph_type)
        # ===== Linear
        W_true = simulate_parameter(g, W_ranges)
        X = simulate_linear_sem(W_true, vals, sem_type)
        # ===== NonLinear
        #X = simulate_nonlinear_sem(g, 1000, 'mim')
        # cov = np.cov(np.transpose(X))
        cov = get_my_normalized_cov(X)
        res_X.append(cov)
        # res_Y.append((W_true != 0).astype('float32'))
        res_Y.append(g.astype('float32'))
    return np.array(res_X), np.array(res_Y)

def gen_train_data_custom_barabasi(ngraphs, vals, nodes, edges, graph_type, sem_type, k):
    # generate graphs
    # for each graph, generate data
    # edge count drawn from uniform(nodes, nodes*6)
    res_X = []
    res_Y = []
    W_ranges = ((-0.5-k, -0.5), (0.5, k+0.5))
    for _ in range(ngraphs):
        g = randomised_barabasi_generator(nodes)
        # ===== Linear
        W_true = simulate_parameter(g, W_ranges)
        X = simulate_linear_sem(W_true, vals, sem_type)
        # ===== NonLinear
        #X = simulate_nonlinear_sem(g, 1000, 'mim')
        # cov = np.cov(np.transpose(X))
        cov = get_my_normalized_cov(X)
        res_X.append(cov)
        # res_Y.append((W_true != 0).astype('float32'))
        res_Y.append(g.astype('float32'))
    return np.array(res_X), np.array(res_Y)

def gen_train_data_k_ensemble(ngraphs, vals, nodes, edges, graph_type, sem_type, k):
    # generate graphs
    # for each graph, generate data
    res_X = []
    res_Y = []
    for _ in range(ngraphs):
        k = np.random.uniform(1,4)
        W_ranges = ((-0.5-k, -0.5), (0.5, k+0.5))
        g = simulate_dag(nodes, edges, graph_type)
        # ===== Linear
        W_true = simulate_parameter(g, W_ranges)
        X = simulate_linear_sem(W_true, vals, sem_type)
        # ===== NonLinear
        #X = simulate_nonlinear_sem(g, 1000, 'mim')
        # cov = np.cov(np.transpose(X))
        cov = get_my_normalized_cov(X)
        res_X.append(cov)
        # res_Y.append((W_true != 0).astype('float32'))
        res_Y.append(g.astype('float32'))
    return np.array(res_X), np.array(res_Y)

def gen_train_data_type_ensemble(ngraphs, vals, nodes, edges, graph_type, sem_type, k):
    # generate graphs
    # for each graph, generate data
    res_X = []
    res_Y = []
    W_ranges = ((-0.5-k, -0.5), (0.5, k+0.5))
    for _ in range(ngraphs):
        g = simulate_dag(nodes, edges, random.choice(("SF", "ER")))
        # ===== Linear
        W_true = simulate_parameter(g, W_ranges)
        X = simulate_linear_sem(W_true, vals, sem_type)
        # ===== NonLinear
        #X = simulate_nonlinear_sem(g, 1000, 'mim')
        # cov = np.cov(np.transpose(X))
        cov = get_my_normalized_cov(X)
        res_X.append(cov)
        # res_Y.append((W_true != 0).astype('float32'))
        res_Y.append(g.astype('float32'))
    return np.array(res_X), np.array(res_Y)

def gen_train_data_notears(ngraphs, vals, nodes, edges, graph_type, sem_type):
    #todo: add parameter k
    res_X = []
    res_Y = []
    for _ in range(ngraphs):
        g = simulate_dag(nodes, edges, graph_type)
        # ===== Linear
        W_true = simulate_parameter(g)
        X = simulate_linear_sem(W_true, vals, sem_type)
        # ===== NonLinear
        #X = simulate_nonlinear_sem(g, 1000, 'mim')
        # cov = np.cov(np.transpose(X))
        res_X.append(X)
        # res_Y.append((W_true != 0).astype('float32'))
        res_Y.append(g.astype('float32'))
    return np.array(res_X), np.array(res_Y)

def split(X, Y):
    mid = int(X.shape[0] * 0.8)
    return X[:mid], Y[:mid], X[mid:], Y[mid:]

def dag_to_cpdag(g):
    dag = cd.DAG.from_amat(g)
    cpdag = dag.cpdag()
    return cpdag.to_amat()[0]

def eval_model(model, x, y):
    out = model(x).numpy()
    out_bin = output_acyclic(out)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits = False)
    loss = bce(out_bin, out).numpy()
    accs_out = np.zeros(7)
    for n in range(out.shape[0]):
    #convert both true and estimated graphs to cpdags
        est_cpdag = dag_to_cpdag(out_bin[n])
        true_cpdag = dag_to_cpdag(y[n])
        accs_out += count_accuracy(est_cpdag, true_cpdag, "list")
    num_dag = sum([is_dag(ypred) for ypred in out_bin])
    fdr = accs_out[2]/out.shape[0]
    prec = accs_out[0]/out.shape[0]
    recall = accs_out[1]/out.shape[0]
    F1 = 2*prec*recall/(prec+recall)
    shd = accs_out[5]/out.shape[0]    
    dag = num_dag/out.shape[0]
    return [fdr, prec, recall, F1, shd, dag]

def count_accuracy(B_true, B_est, data_type = "dict"):
    shd = sum(sum(B_true != B_est))                 
    #number of elements that are not equal

    intersect = np.logical_and((B_est == 1), (B_true == 1))
    num_intersect = sum(sum(intersect))
    if sum(sum(B_est)) == 0:
        prec = 0
    else:
        prec = num_intersect / sum(sum(B_est))
        #intersection divided by number of predicted edges
    if sum(sum(B_true)) == 0:
        recall = 0
    else:
        recall = num_intersect / sum(sum(B_true))
        #intersection divided by number of true edges
    return [prec, recall, 0, 0, 0, shd, 0]

def draw_dag(G):
    #inputs are adjacency matrices
    nx.draw_networkx(cd.DAG.from_amat(G).to_nx())

def output_acyclic(arr):
    #inputs and outputs are numpy arrays of dimension (ngraphs, nodes, nodes)
    #modifies input to output in place
    dim = arr.shape[1]    
    for i, x in enumerate(arr):
        out = np.zeros((dim, dim))
        while True:
            if x.max() > 0.5:
                max_x = x.argmax()//dim
                max_y = np.remainder(x.argmax(), dim)
                x[max_x, max_y] = 0
                out[max_x, max_y] = 1
                if not is_dag(out):
                    out[max_x, max_y] = 0
            else:
                arr[i] = out
                break
    return arr