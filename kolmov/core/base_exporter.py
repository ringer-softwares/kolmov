__all__ = [
    'calc_sp',
    'first_export_tool',
]

import os
import json

import numpy as np
import pandas as pd

from itertools import product 
from Gaugi import load as gload
from Gaugi import save as gsave
from sklearn.metrics import roc_curve
from tensorflow.keras.models import model_from_json

# SP index definition
def calc_sp(pd, fa):
    g_mean = np.sqrt((1-fa) * pd)
    a_mean = ((1-fa) + pd)/2.
    return np.sqrt(g_mean*a_mean)

class first_export_tool(object):
    '''
    This class is the very first export tool to get and prepare the tunings
    to move them for prometheus framework. 

    As in older version of saphyra tunings the thresholds wasn't saved so this class 
    will compute the operation threshold using the data used to train.

    This probably will be changed when saphyra dump the thresholds too.
    '''

    def __init__(self, task_path, operation_dataframe, ringer_data_path, noHAD=False):
        '''
        Arguments:
        - task_path: the path where are all tuned files.
        - operation_dataframe: a .csv file with all models selected to operation.
        - ringer_data: path to ringer_data that can be formated the et and eta.
        - noHAD: noHAD flag to use only EM rings.

        Ex.:
        ringer_data = your_data_path + 'your_file_name...et%i_eta%i.npz'

        '''
        self.task_path         = task_path
        self.op_df            = pd.read_csv(operation_dataframe)
        self.ringer_data_path = ringer_data_path
        self.noHAD            = noHAD
    
    def model_finder(self, tunedfile, model_idx, sort, init):
        '''
        This function will search in the tunedfile for wanted
        model given the model idx, sort and init returning the sequential and the weights.

        Arguments:
        
        - tunedfile: the file with tuned models 
        Ex.: 'tunedDiscr.jobID_0004.pic.gz'

        - model_idx: the model index, this is more convinient when you have many MLP's models.
        - sort: the sort that you want.
        - init: the initialization that you want.
        '''
        for itunedData in tunedfile['tunedData']:
            if ((itunedData['imodel'] == model_idx)\
                and (itunedData['sort'] == sort)\
                and (itunedData['init'] == init)):
                # return the keras sequential and weights
                return itunedData['sequence'], itunedData['weights']
    
    def save_models_dict(self, filename):
        '''
        This function will save the models dictionary obtained in fill_models_dict function.

        Arguments:
        -filename: the name of dictionary to be save.
        '''
        gsave(self.models, '%s.pic.gz' %(filename))

    def fill_models_dict(self):
        '''
        This function will fill the dictionary of operation models using the information
        from the operation dataframe and using the ringer data to calculate the threshold.
        '''
        self.models = {}
        # add the noHAD flag
        self.models['noHAD'] = self.noHAD
        for iet, ieta in product(range(self.op_df.et_bin.nunique()),
                                 range(self.op_df.eta_bin.nunique())):
            print('Processing et bin: %i | eta bin: %i' %(iet, ieta))
            # alocate in the dictionary
            self.models['et%i_eta%i' %(iet, ieta)] = {}

            # 1 step: get the right file, model_id, sort and init to open.
            aux_df = self.op_df.loc[((self.op_df['et_bin'] == iet) &\
                                     (self.op_df['eta_bin'] == ieta)), :].copy()
            f_name, id_model, sort, init = aux_df.iloc[0][['file_name', \
                                                        'model_idx', \
                                                        'sort', \
                                                        'init']].values

            # 2 step: get the model and the weights
            tuned_file = os.path.join(self.task_path, f_name) %(iet, ieta)
            m, w = self.model_finder(tunedfile=gload(tuned_file), model_idx=id_model,
                                sort=sort, init=init)
            # save the model and the weights into the dictionary
            self.models['et%i_eta%i' %(iet, ieta)]['sequential'] = m
            self.models['et%i_eta%i' %(iet, ieta)]['weights']    = w
            # load the keras models in order to get the thresholds
            local_model = model_from_json(json.dumps(m))
            local_model.set_weights(w)

            # 3 step: load the data to predict using the associated model
            print('Load rings data... ')
            data             = gload(self.ringer_data_path %(iet, ieta))
            if self.noHAD: # check if we need all rings or no...
                print('This is a noHAD tuning, using only EM rings...')
                rings            = data['data'][:, 1:89]
            else:
                rings            = data['data'][:, 1:101]
            print('Preprocessing rings...')
            norm             = np.abs(rings.sum(axis=1))
            norm[ norm == 0] = 1 
            rings            = rings/norm[:, None] # normalized rings
            trgt             = data['target']
            # predict on rings
            print('Predict step...')
            nn_output        = local_model.predict(rings)
            print('Getting the operation threshold... ')
            # calculate the thresholds
            fa, pd, thresholds = roc_curve(trgt, nn_output)
            sp = calc_sp(pd, fa)
            # get the knee
            knee = np.argmax(sp)
            # save the threshold
            self.models['et%i_eta%i' %(iet, ieta)]['threshold'] = thresholds[knee]
            print(10*'-')
        print('Done!')
    
