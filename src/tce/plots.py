import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from config import *
import os
import itertools

class plots:

    def __init__(self):

        self.plot_args = [
            {'c': 'red', 'linestyle': '-'},
            {'c': 'green', 'linestyle': '-'},
            {'c': 'blue', 'linestyle': '-'},
            {'c': 'red', 'linestyle': '--'},
            {'c': 'green', 'linestyle': '--'},
            {'c': 'blue', 'linestyle': '--'},
            {'c': 'black', 'linestyle': '-'}
        ]

    def plot(self, MLP_SOLVERS):
        self._plot(MLP_SOLVERS)
        self._plot_confusion_matrix(MLP_SOLVERS)

    def _plot(self, MLP_SOLVERS):

        plt.figure(1)
        legends = []
        i = 0
        for solver in MLP_SOLVERS:
            print("Ploting: %s..." % (solver))
            leg, = plt.plot(
                MLP_SOLVERS[solver]['clfParam']['clf'].loss_curve_,
                label=MLP_SOLVERS[solver]['otherParam']['label']
            )
            legends.append(leg)
            if i > 10:
                break
            i += 1
        plt.grid(True)
        plt.title(MLP_SOLVERS[solver]['clfParam']['hidden_layer_sizes'])
        plt.legend(handles=legends)
        loss_curveplot = PRED_DATA_PATH + solver + '/' + 'loss_curve'
        plt.savefig(loss_curveplot)

        print('Loss Curve has been saved at %s' %(loss_curveplot))

    def _plot_confusion_matrix(self, MLP_SOLVERS):

        for solver in MLP_SOLVERS:
            matrixDf = MLP_SOLVERS[solver]['otherParam']['matrix']
            tcmat = matrixDf[matrixDf.param == 'confusion_matrix']
            cm = tcmat['value'][tcmat['value'].index[0]]
            confusion_matrix = PRED_DATA_PATH + solver + '/' + 'confusion_matrix'
            plt.figure()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('confusion_matrix')
            plt.colorbar()
            tick_marks = np.arange(len(CLASS_LABELS))
            plt.xticks(tick_marks, CLASS_LABELS, rotation=45)
            plt.yticks(tick_marks, CLASS_LABELS)

            normalize = True
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            thresh = cm.max() / 2.0
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, round(cm[i, j], 2),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(confusion_matrix)
