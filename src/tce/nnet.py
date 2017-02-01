import numpy as np
import pandas as pd
# from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from time import time
from config import *
from plots import plots
from sklearn.metrics import f1_score
import os
import sys
import pickle
import pprint

class tceNN:

    def __init__(self):
        # Load TCE transformed data (Principal Components)
        self.tceData = pd.read_csv(
            "%s%s" % (TCE_COMPUTED, TCE_TRANSFORMED_DATA_FNAME)
        )

        #Seperate Principal components data and info parameter to different DFs
        self.tceInfoData = self.tceData[TCE_INFO_COL_LIST]
        self.tceData = self.tceData[
            list(set(self.tceData.columns) - set(TCE_INFO_COL_LIST))
        ]

        # minmaxDf = pd.DataFrame(columns=self.tceData.columns)
        # for col in self.tceData.columns:
            # minmaxDf[col] = pd.DataFrame(MinMaxScaler().fit_transform(self.tceData[col].reshape(-1, 1)))[0]

        # print(minmaxDf.head())
        # print(self.tceData.head())

        # self.tceData = minmaxDf


        self.tceLables = self.tceInfoData[['av_pred_class']]

        self.accuracyDf = pd.DataFrame(columns=['param', 'value'])

        if not os.path.exists(PRED_DATA_PATH):
            os.makedirs(PRED_DATA_PATH)

    def saveOutPutData(self, dirName):
        # Save the training (features and labels) Df
        PRED_DATA_PATH_ = PRED_DATA_PATH + dirName + '/'
        if not os.path.exists(PRED_DATA_PATH_):
            os.makedirs(PRED_DATA_PATH_)

        pd.concat(
            [self.fTraining, self.lTraining],
            axis=1
        ).to_csv(
            "%s%s" % (PRED_DATA_PATH_, TRAINING_DATA_FNAME),
            index=False
        )

        # Save Testing dataset
        testData = pd.concat([self.fTesting, self.lTesting], axis=1)

        testData.to_csv(
            "%s%s" % (PRED_DATA_PATH_, TESTING_DATA_FNAME),
            index=False
        )

        testData['pred'] = self.predictPlanets
        testData.to_csv("%s%s" % (PRED_DATA_PATH_, TESTING_PREDICT_DATA), index=False)
        t = pd.DataFrame(self.predictPlanets)

        self.accuracyDf.to_csv("%s%s" % (PRED_DATA_PATH_, OUTPUT_METRIX), index=False)

        # print(t)

    def computeAccuracyScore(self, y_true, y_pred):
        # http://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score
        ascore = accuracy_score(y_true, y_pred, normalize=True)
        self._append('accuracy_score', ascore)

    def computeConfusionMatrix(self, y_true, y_pred):
        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix
        cmetrix = confusion_matrix(y_true, y_pred)
        self._append('confusion_matrix', cmetrix)

    def computeClassificationReport(self, y_true, y_pred):
        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
        creport = classification_report(y_true, y_pred, target_names=['A', 'B', 'C'])
        self._append('classification_report', creport)


    def _append(self, param, value):
        row = dict()
        row['param'] = param
        row['value'] = value
        self.accuracyDf = self.accuracyDf.append(row, ignore_index=True)


    def splitTrainTest(self):
        self.fTraining, \
            self.fTesting, \
            self.lTraining, \
            self.lTesting = train_test_split(
                self.tceData,
                self.tceLables,
                train_size=0.7,
                random_state=33
            )
        # print(self.lTraining.values.ravel())

    def getClassifier(self, config):
        # check for existing classifier
        if config['otherParam']['trained_classifier']:
            # load the trained classifier
            try:
                print("Loading pre-trained classifier %s" % (config['otherParam']['trained_classifier']))
                print(config['otherParam']['trained_classifier'])
                # pkl_file = open(config['otherParam']['trained_classifier'], 'rb')
                with open(config['otherParam']['trained_classifier'], 'rb') as f:
                    clf = pickle.load(f)
                    pprint.pprint(clf)
                    return ['old', clf]
            except Exception as e:
                print(str(e))
                print("Failed to load the trained classifier %s" % (config['otherParam']['trained_classifier']))
                sys.exit("Exit.")
        else:
            # create a new classifier
            print("Creating a new classifier with parameters %s" % (config['clfParam']))
            clf = MLPClassifier(**config['clfParam'])
            return ['new', clf]

    def trainNN(self, clf):
        print("Training MLPClassifier...")
        start = time()
        clf.fit(self.fTraining, self.lTraining.values.ravel())
        end = time()
        self.timeTrain = round((end - start), 2)
        self._append('trainingTime', self.timeTrain)
        print("Took {:.4f} to train the classifier".format(end - start))
        return clf

    def predictPlanets(self, tclf):
        print("Predicting Planets...")
        start = time()
        self.predictPlanets = tclf.predict(self.fTesting)
        end = time()
        self._append('predictTime', end - start)
        score = tclf.score(self.fTraining, self.lTraining.values.ravel())
        self._append('score', score)
        self._append('TrainingLoss', tclf.loss_)
        self._append('lossCurve', tclf.loss_curve_)
        self.computeAccuracyScore(self.lTesting, self.predictPlanets)
        self.computeConfusionMatrix(self.lTesting, self.predictPlanets)
        self.computeClassificationReport(self.lTesting, self.predictPlanets)

    def saveTrainedClassifier(self, clfStatus, clf, mlp_solver):
        if clfStatus == 'new':
            with open(mlp_solver['otherParam']['new_trained_classifier'], 'wb') as f:
                pickle.dump(clf, f)


if __name__ == '__main__':

    mlps = []

    for solver in MLP_SOLVERS:
        tce = tceNN()
        tce.splitTrainTest()
        clfStatus, clf = tce.getClassifier(MLP_SOLVERS[solver])
        if (clfStatus == 'new'):
            clf = tce.trainNN(clf)
        tce.predictPlanets(clf)
        tce.saveOutPutData(solver)
        tce.saveTrainedClassifier(clfStatus, clf, MLP_SOLVERS[solver])

    #     MLP_SOLVERS[solver]['clf'] = trainedClf
    #     MLP_SOLVERS[solver]['label'] = label

    # plt = plots(len(MLP_SOLVERS))
    # plt.plot(MLP_SOLVERS)
