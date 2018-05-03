import sys
import os 

import numpy as np

sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../networks')))
import networks

sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../isingmodel')))
import isingmodel


class Hopfield(isingmodel.IsingCoupling):
    """Hopfield neural network class.

    See Wikipedia page:
    https://en.wikipedia.org/wiki/Hopfield_network

    """
    def __init__(self, shape, seed=None):
        """Init method.

        Parameters
        ----------
            shape : int tuple
                Shape of the network (e.g. (2,3) or 2). It must satisfy
                np.prod(shape) = nnodes. This parameter needed to plot
                the network. If None, a 1D network (shape = (nnodes, ))
                is assumed. 

            seed : int
                Seed for the pseudo-random number generator used in the
                network evolution.

        """
        # Cast shape to tuple
        try: 
            shape = tuple(shape)
        except TypeError:
            shape = tuple((shape,))

        nnodes = np.prod(shape)
        
        # Initialize list of leanrt patterns
        self.patterns = list()

        # Create empty network
        network = networks.Network(nnodes, weighted=True)

        # Initialize Ising lattice
        isingmodel.IsingCoupling.__init__(
                self, nnodes, network=network, shape=shape, seed=seed)

    @property
    def npatterns(self):
        """Number of learnt patterns.

        """
        return len(self.patterns)


    def learn_pattern_Hebb(self, pattern):
        """Learn pattern using Hebb's rule.

        Parameters
        ----------
            pattern : bool array
                Array with the pattern to be learned. It must have the
                the same shape as the network. True elements are 
                associated with +1 and False with -1. If N-dimensional 
                array is given, it will be flattened.

        """
        if pattern.shape != self.shape:
            # TODO: this could be written in a clearer way
            ValueError("The pattern shape does not match the network one.")

        pattern_flat = pattern.flatten()

        # Convert the bool array to an array with +-1
        pattern_pm = 2*pattern_flat.astype(bool) - 1

        # Update adjacency matrix to learn the pattern
        adjmatrix_change = np.outer(pattern_pm, pattern_pm).astype(float)
        self.network.adjmatrix = np.average(
                [self.network.adjmatrix, adjmatrix_change], axis=0,
                weights=[self.npatterns, 1])


        # Update neighbour lists
        self.update_neighbours()

        # Store the pattern in the patterns list
        self.patterns.append(pattern)
