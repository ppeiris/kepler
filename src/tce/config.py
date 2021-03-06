import ConfigParser
import os

config = ConfigParser.RawConfigParser()
config.read("parameters.conf")

v = config.get('version', 'version')


def appendVersion(fname):
    global v
    if fname:
        s = fname.split('.')
        if (len(s) == 1):
            return fname
        return '.'.join(s[:-1]) + '_' + v + '.' + s[-1]


TCE_COMPUTED = config.get('paths', 'tceDataPath')
if not os.path.exists(TCE_COMPUTED):
    os.makedirs(TCE_COMPUTED)

if not os.path.exists(TCE_COMPUTED + v + '/'):
    os.makedirs(TCE_COMPUTED + v + '/')

TCE_COMPUTED = TCE_COMPUTED + v + '/'


TCE_RAW_DATA_PATH = config.get('paths', 'rawDataPath')
if not os.path.exists(TCE_RAW_DATA_PATH):
    os.makedirs(TCE_RAW_DATA_PATH)

PRED_DATA_PATH = config.get('paths', 'predDataPath')

predPath = PRED_DATA_PATH.split('/')
PRED_DATA_PATH = '/'.join(predPath[:-2]) + '/' + v + '/' + predPath[-2] + '/'
if not os.path.exists(PRED_DATA_PATH):
    os.makedirs(PRED_DATA_PATH)

CLASSIFIER_PATH = config.get('paths', 'trainedClassifierPath')
if not os.path.exists(CLASSIFIER_PATH):
    os.makedirs(CLASSIFIER_PATH)

TCE_RAW_FNAME = config.get('rawdata', 'rawDataFile')
TCE_COL_FNAME = appendVersion(config.get('rawdata', 'colFileName'))
TCE_INFO_COL_LIST = [c[1] for c in config.items("tceInfoCols")]

TCE_USED_COL_LIST_FNAME = appendVersion(config.get('computedFiles', 'tcecollist'))
TCE_RAW_CLEAN_DATA_FNAME = appendVersion(config.get('computedFiles', 'tcerawdata'))
TCE_EIGENVALUES_FNAME = appendVersion(config.get('computedFiles', 'tceeigenvalues'))
TCE_TRANSFORMED_DATA_FNAME = appendVersion(config.get('computedFiles', 'tcepcaTransformedData'))
TCE_PARAMETERS_PC_CORRELATION_FNAME = appendVersion(config.get('computedFiles', 'tceParameterPcCorrelation'))

# # check if the previously trained classifer exists
# CLASSIFER_NAME = config.get('classifier', 'classifierName')
# if not (CLASSIFER_NAME and os.path.isfile(CLASSIFIER_PATH + CLASSIFER_NAME)):
#     CLASSIFER_NAME = None
# else:
#     CLASSIFER_NAME = CLASSIFIER_PATH + CLASSIFER_NAME

TRAINING_DATA_FNAME = appendVersion(config.get('predict', 'trainingdata'))
TESTING_DATA_FNAME = appendVersion(config.get('predict', 'testingdata'))
TESTING_PREDICT_DATA = appendVersion(config.get('predict', 'predictdata'))
OUTPUT_METRIX = appendVersion(config.get('predict', 'predictmetrics'))

CLASS_LABELS = [c[1] for c in config.items("classlabels")]

# MLPClassifier Solvers
MLP_SOLVERS = dict()
for sec in config.sections():
    if 'solver_' in sec:
        solverItems = config.items(sec)
        solver = dict({'clfParam': dict(), 'otherParam': dict()})
        for item in solverItems:
            tmpItem = item[0].split('_')
            if tmpItem[-1] == "float": ## at the end
                if tmpItem[0] == "clf": ## in the begining
                    solver['clfParam']['_'.join(tmpItem[1:-1])] = float(item[1])
                else:
                    solver['otherParam']['_'.join(tmpItem[0:-1])] = float(item[1])

            if tmpItem[-1] == "int": ## at the end
                if tmpItem[0] == "clf": ## in the begining
                    solver['clfParam']['_'.join(tmpItem[1:-1])] = int(item[1])
                else:
                    solver['otherParam']['_'.join(tmpItem[0:-1])] = int(item[1])

            if tmpItem[-1] == "string": ## at the end
                if tmpItem[0] == "clf": ## in the begining
                    solver['clfParam']['_'.join(tmpItem[1:-1])] = item[1]
                else:
                    solver['otherParam']['_'.join(tmpItem[0:-1])] = item[1]

            if tmpItem[-1] == "tuple": ## at the end
                if tmpItem[0] == "clf": ## in the begining
                    solver['clfParam']['_'.join(tmpItem[1:-1])] = tuple([int(i) for i in item[1].split(',')])
                else:
                    solver['otherParam']['_'.join(tmpItem[1:-1])] = tuple([int(i) for i in item[1].split(',')])

        MLP_SOLVERS[sec] = solver


# Check if there a pre trained classifier
for sec in MLP_SOLVERS:
    if 'trained_classifier' in MLP_SOLVERS[sec]['otherParam']:
        tclfpath = CLASSIFIER_PATH + MLP_SOLVERS[sec]['otherParam']['trained_classifier']
        if os.path.isfile(tclfpath):
            MLP_SOLVERS[sec]['otherParam']['trained_classifier'] = tclfpath
        else:
            MLP_SOLVERS[sec]['otherParam']['trained_classifier'] = None
            MLP_SOLVERS[sec]['otherParam']['new_trained_classifier'] = tclfpath
    else:
        MLP_SOLVERS[sec]['otherParam']['trained_classifier'] = None
        MLP_SOLVERS[sec]['otherParam']['new_trained_classifier'] = tclfpath
    # load the default values in case the hidden_layer_sizes not defined in the param file
    if not 'hidden_layer_sizes' in MLP_SOLVERS[sec]['clfParam']:
        MLP_SOLVERS[sec]['clfParam']['hidden_layer_sizes'] = (100, )


if __name__ == '__main__':

    print PRED_DATA_PATH
    # print [c[1] for c in config.items("classlabels")]
    # for i in MLP_SOLVERS:
    #     MLP_SOLVERS[i]['clf'] = 'test'
    #     # print MLP_SOLVERS[i]['clf']

    # print(MLP_SOLVERS)
