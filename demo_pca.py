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
        description="Demo for embedding data via PCA",
        parents=parents
        )
    
    parser.add_argument(
        '--data', '-X',
        metavar='FILEPATH',
        help="The filename of a csv with datapoints.",
        default=None
        )
    
    return parser


def plot2D_kmeans(ax, X, Y, means):
    ''' plot2D_kmeans(ax, X, Y, means)
    Plots the KMeans status to given axes.
    
    Args:
        ax: The matplotlib axes.
        X: The ground truth dataset.
        Y: The assigned cluster labels.
        means: The cluster centers.
    '''
    ax.scatter(X[:,0], X[:,1], s=1, c=Y)
    ax.scatter(means[:,0], means[:,1], s=100, c='r', marker='x')
    
    
def plot_convergence(ax, step, delta):
    ''' plot_convergence(ax, step, delta)
    Plots the convergence of the KMeans process
    based on the update delta.
    
    Args:
        ax: The matplotlib axes.
        step: The update step number.
        delta: The update delta.
    '''
    ax.scatter(step, delta, c='r')
    ax.set_xlabel('step')
    ax.set_ylabel('delta')


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
        "    data ('circle.txt'): ",
        default='circle.txt'
        )
    
    args.centers = args.centers if args.centers else myinput(
        "The number of cluster centers.\n" +
        "    centers (2): ",
        default=2,
        cast=int
        )
    
    args.epsilon = args.epsilon if args.epsilon else myinput(
        "The convergenc threshold.\n" +
        "    epsilon (0): ",
        default=0.0,
        cast=float
        )
    
    args.mode = args.mode if args.mode else myinput(
        "Choose an initialization mode.\n    > " +
        "\n    > ".join(KMEANS_INIT_MODES) +
        "\n    mode (select): ",
        default='select',
        cast=lambda m: m if m in KMEANS_INIT_MODES else raise_(ValueError("Unknown mode"))
        )
    
    args.kernel = args.kernel if args.kernel else myinput(
        "Choose a kernel for kernel KMeans.\n    > " +
        "\n    > ".join(kernel.__all__) +
        "\n    kernel (none): ",
        default='none',
        cast=lambda k: k if k in kernel.__all__ else raise_(ValueError("Unknown kernel"))
        )
    
    if args.kernel != 'none':
        args.params = args.params if args.params else myinput(
            "Parameters for the kernel, if required.\n" +
            "    params (gamma=1.0): ",
            default='gamma=1.0'
            )
        args.params = kernel.parse_params(args.params)
        kernel_func = getattr(kernel, args.kernel)
    else:
        kernel_func = None
    
    args.video = args.video if args.video else myinput(
        "The filename for video record.\n" + 
        "    video (None): ",
        default=None
        )
    
    #Load data
    print("\nLoad data...")
    GT = np.genfromtxt(args.data, delimiter=',')
    
    if kernel_func:
        print("\nInit kernel with:")
        print(args.params)
        X = kernel_func(GT, GT, args.params)
    else:
        X = GT
        

    #Init KMeans
    print("\nInit means with mode: {}".format(args.mode))
    means = init_kmeans(X, args.centers, args.mode)
    
    #Run KMeans
    fig, axes = arrange_subplots(2)
    axes[1].title.set_text("Convergence")
    
    print("\nCompute KMeans...")
    def plot_update(fargs):
        Y = fargs[0]
        means = fargs[1]
        delta = fargs[2]
        step = fargs[3]
        print("Render step: {}".format(step))
        axes[0].clear()
        plot2D_kmeans(axes[0], GT, Y, means)
        axes[0].title.set_text("KMeans step: {}".format(step))
        plot_convergence(axes[1], step, np.sqrt(delta))
    
    if args.video:
        import os
        import matplotlib
        import matplotlib.animation as ani
        
        dir = os.path.dirname(os.path.realpath(args.video))
        if not os.path.isdir(dir):
            os.mkdir(dir)
        
        Writer = ani.writers['ffmpeg'](fps=1, metadata=dict(artist='Gerald Baulig'), bitrate=1800)
        run_kmeans = lambda: kmeans(X, means, args.epsilon)
        plot_ani = ani.FuncAnimation(fig, plot_update, run_kmeans, interval=5000)
        plot_ani.save(args.video, writer=Writer)
        print("\nSave video to {}".format(args.video))
    else:
        for Y, means, delta, step in kmeans(X, means, args.epsilon): #Extract update steps of the generator
            if plt.fignum_exists(fig.number):
                plot_update((Y, means, delta, step))
                plt.show(block=False)
                plt.pause(0.1) #give a update pause
            else:
                return 1
        print("\nDone!")
        plt.show() #stay figure
    return 0

if __name__ == '__main__':
    parser = init_argparse()
    args, _ = parser.parse_known_args()
    main(args)
    