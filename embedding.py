'''


Author: Gerald Baulig
'''

#Global libs
import numpy as np


def pca(X, dim=2):
    '''
    Runs PCA on the NxD array X in order to reduce its dimensionality to
    dim dimensions.
    
    Agrs:
        X: The dataset.
        dim: The number of dimensions to be reduced at.
    Returns:
        x: The reduced representations.
        eigvec: The eigenvectors of the scatter matrix.
        eigval: The eigenvalues of the scatter matrix.
        S: The scatter matrix.
        M: The centralized dataset.
        m: The mean of the dataset.
    '''
    #guided by https://sebastianraschka.com/Articles/2014_pca_step_by_step.html
    m = np.mean(X.T, axis=1)
    M = (X - m).T # Centralize the data.
    S = np.cov(M) # The Scatter Matrix
    eigval, eigvec = np.linalg.eigh(S) # get the covariance
    idx = np.argsort(eigval)[::-1] # sorting the eigenvalues to get k first largest eigenvectors
    
    eigvec = eigvec[:,idx].T # apply sorting idx
    eigval = eigval[idx]
    
    x = np.dot(eigvec[:dim], M).real.T # project the data in the eigenspace
    return x, eigvec, eigval, S, M, m


def lda(X, Y, dim=2):
    '''
    Runs LDA on the NxD array X in order to reduce its dimensionality to
    dims dimensions.
    
    Args:
        X: The dataset.
        Y: The labels.
        dim: The number of dimensions to be reduced at.
    Returns:
        x: The reduced representations.
        eigvec: The eigenvectors of the scatter matrix.
            (W = eigvec[:dim])
    '''
    C = np.unique(Y) #classes
    A = np.zeros((X.shape[0],len(C))) #association matrix
    for idx, c in enumerate(C):
        A[Y==c,idx] = 1
    N = np.sum(A, axis=0) #num of associations

    xm = np.dot(X.T, A) / N
    tm = np.mean(X, axis=0)
    mm = xm - tm[:,None]
    SB = np.dot(mm * N, mm.T)
    W = X.T - np.dot(xm, A.T)
    SW = np.zeros(SB.shape)
    for idx, c in enumerate(C):
        w = W[:,Y==c]
        SW += np.dot(w, w.T) / N[idx]

    try:
        SW = np.linalg.inv(SW)
    except:
        print("Warning: Fallback to pseudo inverse!")
        SW = np.linalg.pinv(SW)

    eigval, eigvec = np.linalg.eigh(np.dot(SW, SB))
    idx = np.argsort(eigval)[::-1]
    eigvec = eigvec[:,idx]
    w = eigvec[:,:dim]
    x = np.dot(X, w).real
    return x, eigvec


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


def x2p_asym(X, tol=1e-5, perplexity=30.0):
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
    D = (-2 * np.dot(X, X.T) + sum_X).T + sum_X
    P = np.zeros((n,n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):
        # Print progress
        if (i+1) % 500 == 0:
            print("Computed %d of %d P-values." % (i+1, n))

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


def x2p_sym(X, tol=1e-5, perplexity=30.0):
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
    D = (-2 * np.dot(X, X.T) + sum_X).T + sum_X
    P = np.zeros((n,n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):
        # Print progress
        if (i+1) % 500 == 0:
            print("Computed %d of %d P-values." % (i+1, n))

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


def tsne(X,
    dim=2,
    tol=1e-5,
    perplexity=30.0,
    exag=4.0,
    min_p=1e-12,
    mom_0=0.5,
    mom_n=0.8,
    min_gain=0.01,
    eta=500,
    epsilon=1.0,
    max_iter=1000,
    sym=False
    ):
    '''
    Runs t-SNE on the dataset in the NxD array X to reduce its
    dimensionality to dims dimensions.
    '''

    # Initialize variables
    (n, d) = X.shape
    x = np.random.randn(n, dim)
    dx = np.zeros((n, dim))
    ix = np.zeros((n, dim))
    gains = np.ones((n, dim))

    # Compute P-values
    P = x2p_sym(X, tol, perplexity) if sym else x2p_sym(X, tol, perplexity)
    P = P + P.T
    P = P / np.sum(P)
    P = P * exag # early exaggeration. Lying about P?
    P = np.maximum(P, min_p)

    # Run iterations
    for step in range(max_iter):
        # Compute pairwise affinities
        sum_x = np.sum(np.square(x), axis=1)
        num = -2.0 * np.dot(x, x.T)
 
        if sym:
            num = np.exp(-1.0 * ((num + sum_x).T + sum_x))
        else:
            num = 1.0 / (1.0 + (num + sum_x).T + sum_x)
        
        num[range(n), range(n)] = 0.0 #eliminate D
        Q = num / np.sum(num)
        Q = np.maximum(Q, min_p)

        # Compute gradient
        PQ = P - Q
        if sym:
            for i in range(n):
                dx[i, :] = np.sum(np.tile(PQ[:,i], (dim, 1)).T * (x[i,:] - x), 0)
        else:
            for i in range(n):
                dx[i, :] = np.sum(np.tile(PQ[:,i] * num[:,i], (dim,1)).T * (x[i,:] - x), 0)

        # Perform the update
        if step < 20:
            mom = mom_0
        else:
            mom = mom_n
        mask = (dx > 0.0) == (ix > 0.0)
        gains = (gains + 0.2) * ~mask + (gains * 0.8) * mask # TODO: what's about these constants?!
        gains[gains < min_gain] = min_gain
        ix = mom * ix - eta * (gains * dx)
        x = x + ix
        x = x - np.tile(np.mean(x, axis=0), (n, 1))
         
        if step == 100: # Stop lying about P-values. Please what?
            P = P / exag
        
        err = np.sum(P * np.log(P / Q))

        yield x, err, P, Q, step # Yield solution
        
        if err <= epsilon: # Terminate on convergation
            break
        
        