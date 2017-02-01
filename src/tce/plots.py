import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from config import *
import os


class plots:

    def __init__(self, numPlots):

        self.plot_args = [
            {'c': 'red', 'linestyle': '-'},
            {'c': 'green', 'linestyle': '-'},
            {'c': 'blue', 'linestyle': '-'},
            {'c': 'red', 'linestyle': '--'},
            {'c': 'green', 'linestyle': '--'},
            {'c': 'blue', 'linestyle': '--'},
            {'c': 'black', 'linestyle': '-'}
        ]

        self.nrows = int(numPlots)/2
        self.ncols = 2
        # self.plt = plt.figure(1)
        # self.fig, self.axes = plt.subplots(self.nrows, self.ncols, figsize=(15, 10))

    def plot(self, MLP_SOLVERS):
        plt.figure(1)
        legends = []
        i = 0
        for MLP in MLP_SOLVERS:
            print("Ploting: %s" % (MLP))
            leg, = plt.plot(MLP_SOLVERS[MLP]['clf'].loss_curve_, label=MLP_SOLVERS[MLP]['label'])
            legends.append(leg)
            if i > 10:
                break
            i += 1
        plt.legend(handles=legends)
        plt.savefig('new_7')
