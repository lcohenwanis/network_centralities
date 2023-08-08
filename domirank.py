"""DomiRank centrality."""

import networkx as nx

__all__ = ["domirank", "domirank_power_iter"]


@nx._dispatch
def domirank(G: nx.Graph, sigma: float = 0.0000000001, theta: float = 1) -> dict:
    """Returns the DomiRank centrality of each of the nodes in the graph.

    DomiRank quantifies the dominance of the networks’
    nodes in their respective neighborhoods.

    Parameters:
    ----------
     G : graph
      A NetworkX graph.

    sigma: float
        competition factor bounded between (0, -1/λ_N),
        where λ_N represents the minimum eigenvector of A.

    theta: float, optional
        optional weight parameter to scale the domiranks (from original equation in reference paper).

    Returns
    -------
    domirank : dictionary
       Dictionary of nodes with DomiRank as value.

    Examples
    --------
    >>> G = nx.erdos_renyi_graph(n=5, p=0.5)
    >>> sigma = 0.5
    >>> domi = nx.domirank(G, sigma)


     References
    ----------
    .. [1] M. Engsig, A. Tejedor, Y. Moreno, E. Foufoula-Georgiou, C. Kasmi,
       "DomiRank Centrality: revealing structural fragility of complex networks via node dominance."
       https://arxiv.org/abs/2305.09589
    """
    import numpy as np
    import scipy as sp

    # Get number of nodes
    N = len(G)
    if N == 0:
        return {}

    # get Adjacency matrix
    A = nx.adjacency_matrix(G).todense()

    # compute λ_N - smallest eigenvalue of A
    e = np.linalg.eigvals(A)
    lambda_N = min(e)

    # Verify the given sigma value is viable
    try:
        # check if sigma is between (0, -1/λ_N)
        assert sigma > 0
        assert sigma < -1 / lambda_N

    except Exception as e:
        print(f"Sigma is out of bounds. Needs to be between 0 and {-1 / lambda_N}")

    # If sigma is too high, set it to just under -1 / lambda
    if sigma > -1 / lambda_N:
        sigma = -1 / lambda_N + 0.0000000001

    # Verify that sigma * A + np.identity(N) is invertible
    try:
        assert np.linalg.det(sigma * A + np.identity(N)) != 0

    except Exception as e:
        print(
            "One assumptions of DomiRank isn't met: The determinent of sigma * A + np.identity(N) is not invertible."
        )

    # DomiRank Centrality calculation
    dr = (
        theta
        * sigma
        * np.dot(
            np.linalg.inv(sigma * A + np.identity(N)),
            np.dot(A, np.ones((N, 1))),
        )
    )

    # Converts output to dictionary with key: node, value: centrality
    dr_dict = {i: dr[i][0] for i in range(len(dr))}

    return dr_dict

def domirank_power_iter(G: nx.Graph,
            sigma: float,
            theta: float = 1,
            max_iter: int = 100,
            tol: float = 1.0e-6,
            nstart: dict = None,
            weight: str = "weight",
            dangling: dict = None,
            ) -> dict:
    """Returns the DomiRank centrality of each of the nodes in the graph.

    DomiRank quantifies the dominance of the networks’
    nodes in their respective neighborhoods. This power iteration
    implementation is very similar pagerank.

    Parameters:
    ----------
     G : graph
      A NetworkX graph.

    sigma: float
        competition factor bounded between (0, -1/λ_N),
        where λ_N represents the minimum eigenvector of A.

    theta: float, optional
        optional weight parameter to scale the domiranks ("threshold for domination" in reference paper).

    max_iter : integer, optional
        Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
      Error tolerance used to check convergence in power method solver.
      The iteration will stop after a tolerance of ``len(G) * tol`` is reached.

    nstart : dictionary, optional
      Starting value of DomiRank iteration for each node.

    weight : key, optional
        Edge data key to use as weight.  If None weights are set to 1.

    dangling: dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified) This must be selected to result in an irreducible transition
      matrix (see notes under google_matrix). It may be common to have the
      dangling dict to be the same as the personalization dict.


    Returns
    -------
    domirank : dictionary
       Dictionary of nodes with DomiRank as value.

    Examples
    --------
    >>> G = nx.erdos_renyi_graph(n=5, p=0.5)
    >>> sigma = 0.5
    >>> domi = nx.domirank(A, sigma)


     References
    ----------
    .. [1] M. Engsig, A. Tejedor, Y. Moreno, E. Foufoula-Georgiou, C. Kasmi,
       "DomiRank Centrality: revealing structural fragility of complex networks via node dominance."
       https://arxiv.org/abs/2305.09589
    """
    import numpy as np
    import scipy as sp

    # Get number of nodes
    N = len(G)
    if N == 0:
        return {}

    # get Adjacency matrix
    nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)
    S = A.sum(axis=1)
    S[S != 0] = 1.0 / S[S != 0]
    # TODO: csr_array
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A = Q @ A

    # initial vector
    if nstart is None:
        x = np.repeat(1.0 / N, N)
    else:
        x = np.array([nstart.get(n, 0) for n in nodelist], dtype=float)
        x /= x.sum()

    # Dangling nodes -
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = np.where(S == 0)[0]



    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        # x = (sigma * A) @ (theta * np.ones(N) - xlast)
        x = (sigma * A) @ (theta * np.ones(N) - sum(x[is_dangling]) * dangling_weights)
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum()
        if err < N * tol:
            return dict(zip(nodelist, map(float, x)))
    raise nx.PowerIterationFailedConvergence(max_iter)
