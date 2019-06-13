#!/usr/bin/env python
'''


Author: Gerald Baulig
'''

#Global libs
from argparse import ArgumentParser
import matplotlib.pyplot as plt

#Local libs
from embedding import *
from utils import *


def init_argparse(parents=[]):
    ''' init_argparse(parents=[]) -> parser
    Initialize an ArgumentParser for this module.
    
    Args:
        parents: A list of ArgumentParsers of other scripts, if there are any.
        
    Returns:
        parser: The ArgumentParsers.
    '''
    parser = ArgumentParser(
        description="Demo for embedding data via t-SNE",
        parents=parents
        )
    
    parser.add_argument(
        '--data', '-X',
        metavar='FILEPATH',
        help="The filename of a csv with datapoints.",
        default=None
        )
    
    parser.add_argument(
        '--labels', '-Y',
        metavar='FILEPATH',
        help="The filename of a csv with labels.",
        default=None
        )
    
    return parser


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
    args.data = args.data if args.data else myinput(
        "The filename of a csv with datapoints.\n" + 
        "    data (mnist_X.csv): ",
        default='mnist_X.csv'
        )
    
    args.labels = args.labels if args.labels else myinput(
        "The filename of a csv with labels.\n" + 
        "    data (mnist_label.csv): ",
        default='mnist_label.csv'
        )
    
    #Load data
    print("\nLoad data...")
    X = np.genfromtxt(args.data, delimiter=',')
    Y = np.genfromtxt(args.labels, delimiter=',')
    
    print("\nCompute t-SNE...")
    X, _, _, _, _, _ = pca(X)
    
    for x, err, step in tsne(X):
        # Compute current value of cost function
        plt.clf()
        plt.title("t-SNE step {}: Error {}".format(step, err))
        plt.scatter(x[:,0], x[:,1], s=1, c=Y)
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.01)
        if step % 5 == 0:
            plt.savefig('media/t-SNE_asym_{0:04d}.png'.format(step))
        if not plt.fignum_exists(1):
            return 1
    
    print("\nDone!")
    plt.show()
    
    return 0

if __name__ == '__main__':
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    exit(main(args))
    