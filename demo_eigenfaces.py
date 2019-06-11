#!/usr/bin/env python
'''


Author: Gerald Baulig
'''

#Global libs
import os
from glob import iglob
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import cv2

#Local libs
from embedding import *
from utils import *


SD = os.path.dirname(os.path.realpath(__file__))
DEFAULT_ROOT = SD + '/att_faces'


def init_argparse(parents=[]):
    ''' init_argparse(parents=[]) -> parser
    Initialize an ArgumentParser for this module.
    
    Args:
        parents: A list of ArgumentParsers of other scripts, if there are any.
        
    Returns:
        parser: The ArgumentParsers.
    '''
    parser = ArgumentParser(
        description="Demo for embedding faces via PCA",
        parents=parents
        )
    
    parser.add_argument(
        '--root', '-R',
        metavar='DIR',
        help="The root folder ORL face dataset.",
        default=None
        )
    
    return parser


def load_dataset(root):
    X = []
    Y = []
    for y, cdir in enumerate(iglob(root + '/s*/')):
        print("Process folder: {}".format(cdir))
        for imfile in iglob(cdir + '*.pgm'):
            print("Process image: {}".format(imfile))
            face = np.array(cv2.imread(imfile, 0)).flatten()
            X.append(face)
            Y.append(y)
    X = np.array(X).real
    Y = np.array(Y).real
    return X, Y


def img_tiling(X, shape):
    N = X.shape[0]
    w = shape[1]
    h = shape[0]
    cols = int(np.floor(np.sqrt(N)))
    rows = int(np.ceil(N/cols))
    tiles = np.zeros((rows*h, cols*w))
    
    for r in range(rows):
        for c in range(cols):
            n = r+c
            x = np.reshape(X[n,:], shape)
            tiles[r*h:(r+1)*h, c*w:(c+1)*w] = x
            
    return tiles


def main(args):
    ''' main(args) -> exit code
    The main function to execute this script.
    
    Args:
        args: The namespace object of an ArgumentParser.
    Returns:
        An exit code. (0=OK)
    '''
    
    #Validate input. If not given switch to interactive mode!
    print("Validate input...")
    args.root = args.root if args.root else myinput(
        "The root folder ORL face dataset.\n" + 
        "    root ({}): ".format(DEFAULT_ROOT),
        default=DEFAULT_ROOT
        )
    
    #Load data
    print("\nLoad data...")
    X, Y = load_dataset(args.root)
    
    print("Loaded data shape: {}".format(X.shape))
    print("Loaded label shape: {}".format(Y.shape))
    
    print("\nCompute PCA...")
    x, eigvec, eigval, _, _, m = pca(X)
    
    print("\nPlot the result...")
    plt.figure(1)
    plt.title("PCA Faces: Final Result")
    plt.scatter(x[:,0], x[:,1], s=5, c=Y)
    plt.colorbar()
    
    plt.figure(2)
    plt.axis('off')
    plt.title("Eigenfaces")
    tiles = img_tiling(eigvec[:25], (112,92))
    plt.imshow(tiles)
    
    plt.figure(3)
    plt.axis('off')
    plt.title("Reconstruction")
    recons = np.dot(eigvec[:10].T, eigval[:10]) + m
    recons = img_tiling(recons, (112,92))
    plt.imshow(recons)
    
    print("\nDone!")
    plt.show()
    return 0

if __name__ == '__main__':
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    exit(main(args))
    