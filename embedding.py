'''


Author: Gerald Baulig
'''

#Global libs
import numpy as np


def Hbeta(D, beta=1.0):
    '''
    Compute the perplexity and the P-row for a specific value of the
    precision of a Gaussian distribution.
    
    Args:
    
    Returns:
        
    '''

    # Compute P-row and corresponding perplexity
    P = np.exp(-D * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X, tol=1e-5, perplexity=30.0):
    '''
    Performs a binary search to get P-values in such a way that each
    conditional Gaussian has the same perplexity.
    
    Args:
    
    Returns:
    '''

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):
        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X, dims=2):
    '''
    Runs PCA on the NxD array X in order to reduce its dimensionality to
    dims dimensions.
    '''
    #guided by https://sebastianraschka.com/Articles/2014_pca_step_by_step.html
    M = (X - np.mean(X.T, axis=1)).T # Centralize the data. sum((x-m)(x-m)^T)
    S = np.cov(M) # The Scatter Matrix: 
    eigval, eigvec = np.linalg.eig(S) #get the covariance
    idx = np.argsort(eigval)[::-1] # sorting the eigenvalues in ascending order to get k first largest eigenvectors
    
    eigvec = eigvec[:,idx] # apply sorting idx
    eigval = eigval[idx]
    
    Y = np.dot(eigvec[:,:dims].T, M) # project the data in the eigenspace
    return Y, eigvec, eigval


def lda(X, dims=2):
    '''
    Runs LDA on the NxD array X in order to reduce its dimensionality to
    dims dimensions.
    '''
    pass


def tsne(X, dims=2, perplexity=30.0,):
    '''
    Runs t-SNE on the dataset in the NxD array X to reduce its
    dimensionality to dims dimensions. The syntaxis of the function is
    `Y = tsne.tsne(X, dims, perplexity), where X is an NxD NumPy array.
    '''

    # Initialize variables
    X = pca(X, X.shape[1]).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, dims)
    dY = np.zeros((n, dims))
    iY = np.zeros((n, dims))
    gains = np.ones((n, dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y