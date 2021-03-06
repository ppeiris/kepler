import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import *

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 18)
pd.set_option('display.width', 500)


class dataCls:
    """ Load data from flat files

    Load data files from flat files and clean them up

    """

    # Number of principal components to consider after PCA. This will determine
    # after analysis of cumulative variance proportions
    nPC = 20

    def __init__(self, colFile, dataFile):
        """
        Load Column data and clean up them
        """
        colsDf = pd.read_table(
            "%s%s" % (TCE_RAW_DATA_PATH, colFile),
            comment="#"
        )

        tceCols = colsDf.rename(columns={
            colsDf.columns[0]: 'data'
        })

        tceCols['parameter'] = tceCols.data.apply(lambda x: x.split(':')[0])
        tceCols['description'] = tceCols.data.apply(lambda x: x.split(':')[1])

        # Find units from the columns lists
        def __findTceUnits(val):
            sval = val.split('[')
            if sval[-1][-1] == ']':
                return sval[-1][:-1]
        # find units
        tceCols['units'] = tceCols.description.apply(__findTceUnits)
        # Description of the parameter
        tceCols['description'] = tceCols.description.apply(
            lambda x: x.split('[')[0]
        )
        # Parameter
        tceCols['parameter'] = tceCols.parameter.apply(
            lambda x: x.split('COLUMN ')[1]
        )

        del tceCols['data']
        del colsDf

        self.tceCols = tceCols

        # Save TCE Column data
        tceColListFile = "%s%s" % (TCE_COMPUTED, TCE_USED_COL_LIST_FNAME)
        # Save process raw data
        self.tceCols.to_csv(
            tceColListFile,
            index=False
        )

        """
        Load Threshold Crossing Event (TCE) data
        """
        self.tceRawDf = pd.read_csv(
            "%s%s" % (TCE_RAW_DATA_PATH, dataFile),
            comment="#",
            usecols=list(self.tceCols.parameter)
        )

        self.tceInfoData = self.tceRawDf[TCE_INFO_COL_LIST]
        self.tceRawDf = self.tceRawDf[
            list(set(self.tceRawDf.columns) - set(TCE_INFO_COL_LIST))
        ]

        # Drop all the columns that are empty (NaN)
        self.tceRawDf = self.tceRawDf.dropna(axis='columns', how='all')

        # Remove any column with all zeros
        self.tceRawDf = self.tceRawDf.loc[:, (self.tceRawDf != 0.0).any(axis=0)]
        self.n_rows_rawData, self.n_cols_rawData = self.tceRawDf.shape
        self._standardizeData()

        tceRawFile = "%s%s" % (TCE_COMPUTED, TCE_RAW_CLEAN_DATA_FNAME)
        pd.concat(
            [self.tceInfoData, self.tceRawDf],
            axis=1
        ).to_csv(
            tceRawFile,
            index=False
        )

        print("TCE raw data saved at : %s" % (tceRawFile))

        # Save first 1000 records to a file
        self.tceRawDf[:1000].to_csv(
            "%s%s" % (TCE_COMPUTED, "tce_1000.csv"),
            index=True
        )

    def _standardizeData(self):
        """
        @brief      Standardize data
                    http://scikit-learn.org/stable/modules/preprocessing.html
        """
        stdScaler = StandardScaler().fit(self.tceRawDf)
        self.tceData = pd.DataFrame(
            stdScaler.transform(self.tceRawDf),
            columns=list(self.tceRawDf.columns)
        )
        self.n_rows, self.n_cols = self.tceData.shape
        self.tceData[:1000].to_csv(
            "%s%s" % (TCE_COMPUTED, "tce_std1000.csv"),
            index=True
        )

    def pcaTce(self):
        """
        @brief      Apply Principal Component Analysis to TCE Data
        """
        self.pca = PCA(
            n_components=self.n_cols
        )

        self.pca.fit(self.tceData)

        # Eigenvalues and its proportions
        self.eigenDf = pd.DataFrame({
            'Component': ["PC%s" % (i) for i in range(1, self.n_cols + 1)],
            'Eigenvalue': self.pca.explained_variance_,
            'Proportion (%)': self.pca.explained_variance_ratio_.round(4) * 100
        })

        self.eigenDf['Cumulative (%)'] = 0
        cu = 0.0
        for i in self.eigenDf.index:
            cu += self.eigenDf.iloc[i]['Proportion (%)']
            self.eigenDf.loc[i, 'Cumulative (%)'] = cu

        fileEig = "%s%s" % (TCE_COMPUTED, TCE_EIGENVALUES_FNAME)
        print("Eigenvalues are saved at: %s" % (fileEig))
        self.eigenDf.to_csv(
            fileEig,
            index=False
        )

        # Transform data
        self.tceTransData = pd.DataFrame(self.pca.transform(self.tceData))

        if len(self.tceTransData.columns) < self.nPC:
            print "Reduce Principal Components %s to %s" % (self.nPC, len(self.tceTransData.columns))
            self.nPC = len(self.tceTransData.columns)

        self.tceTransData = self.tceTransData[[i for i in range(0, self.nPC)]]

        pcCols = [
            [i, "PC%s" % (i)] for i in range(0, self.nPC)
        ]

        self.tceTransData = self.tceTransData.rename(
            columns=dict(pcCols)
        )

        # save transformed data to a file
        tceTransDataFile = '%s%s' % (TCE_COMPUTED, TCE_TRANSFORMED_DATA_FNAME)
        pd.concat(
            [self.tceInfoData, self.tceTransData],
            axis=1
        ).to_csv(
            tceTransDataFile,
            index=False
        )

    def pcaInterpretation(self):
        """
        @brief      Find out what parameters are contributing to
                    Principal Componenets
        """

        # Combine both raw and tranformed data in to a one DF
        interpreData = pd.concat(
            [self.tceData, self.tceTransData],
            axis=1
        )

        interpreDataCorr = interpreData.corr()
        # print(interpreDataCorr)
        pcCols = ["PC%s" % (i) for i in range(0, self.nPC)]
        interpreDataCorr = interpreDataCorr[pcCols]
        interpreDataCorr.to_csv(
            "%s%s" % (TCE_COMPUTED, TCE_PARAMETERS_PC_CORRELATION_FNAME),
            index=True
        )
        # print(interpreData.head())
        # print(interpreDataCorr[:len(pcCols)])


if __name__ == '__main__':
    tcedata = dataCls(TCE_COL_FNAME, TCE_RAW_FNAME)
    tcedata.pcaTce()
    tcedata.pcaInterpretation()
