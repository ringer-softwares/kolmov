__all__ = [
    'numpy_to_df'
]

import os
import json

import numpy as np
import pandas as pd

from itertools import product
from tensorflow.keras.models import model_from_json
from Gaugi import Logger
from Gaugi.messenger.macros import *
from Gaugi import load as gload



def weights_list_to_array(w_list):
    '''
    A function to transform the weigths from list to array.
    '''
    n_list = []
    for iweigths in w_list:
        n_list.append(np.array(iweigths))
    return n_list



#
# Class numpy_to_df
#
class numpy_to_df(Logger):

    def __init__(self, raw_data):
        '''
        This class has the objective of transform the ringer data into pandas Dataframe
        in order to make easer to manipulate and extract information.

        Arguments:
        - raw_data: the file .npz created in prometheus framework
        Ex.:
        my_data_path =data18_path  some_path_to_data/meaninful_data_name_et%i_eta%i.npz')
        et, eta      = 2, 0
        my_raw_data = dict(np.load(my_data_path %(et, eta)))
        '''
        Logger.__init__(self)
        self.raw_data      = raw_data
        self.et_bin        = raw_data['etBinIdx'].tolist()
        self.eta_bin       = raw_data['etaBinIdx'].tolist()
        # create a pandas DataFrame
        MSG_INFO(self, 'Creating a pandas Dataframe... ')
        self.df_           = pd.DataFrame(data=self.raw_data['data'],
                                          columns=self.raw_data['features'])
        self.df_['target'] = self.raw_data['target']

    def get_df(self):
        '''
        This method return the pandas Dataframe.
        '''
        return self.df_

    def get_rings(self, noHAD=False):
        '''
        This method will return a pandas Dataframe with the rings.
        The flag noHAD is used to remove the hadronic rings.

        Arguments:
        - noHAD: a flag to remove or not the hadronic rings.
        '''
        # decide if want hadronic rings or not
        if noHAD:
            return self.df_[['L2Calo_ring_%i' %(i) for i in range(88)]].copy().values
        else:
            return self.df_[['L2Calo_ring_%i' %(i) for i in range(100)]].copy().values

    def add_tuning_decision(self, tuned_models_dict, jpsiee=True):
        '''
        This method will add into the Dataframe the decision from a given tuning.

        The tuned_models_dict can be more the one tuning. The key in the dictionary
        will be the key of the decision in the Dataframe.

        Arguments:
        - tuned_models_dict: the dictionary which contains the tunings that you want add
        the decision.
        Ex.:
        my_tuned_modes = {
            'model_number1' : (path_to_tuning, path_to_threshold)
        }
        - jpsiee: use the jpsiee et and eta bins
        '''
        if jpsiee:
            aux_index =  list(product(range(3), range(5))).index((self.et_bin, self.eta_bin))
        else:
            aux_index =  list(product(range(5), range(5))).index((self.et_bin, self.eta_bin))
        # FIXME: need to adapt the code to load the new exported tunings
        # loop over the items to build a model dict
        for ikey, (ituning_file, ithr_file) in tuned_models_dict.items():
            MSG_INFO(self, 'Adding %s model to Dataframe' %(ikey))
            with open(ituning_file) as f1, open(ithr_file) as f2:
                tuning_, thr_ = json.load(f1), json.load(f2)
            # get the exact model
            sequential, weights = (tuning_['models'][aux_index]['sequential'],
                                   weights_list_to_array(tuning_['models'][aux_index]['weights']))

            m_thr    = thr_['thresholds'][aux_index]['threshold']
            if m_thr[0] != 0.0:
                print('Not implemented yet!')
                break
            else:
                thr = m_thr[-1]
                print('For %s using threshold: %1.4f \n' %(ikey,thr))
            # build the model
            local_model    = model_from_json(json.dumps(sequential))
            local_model.set_weights(weights)
            print(local_model.summary())
            print('\n')
            # get rings
            rings          = self.get_rings()
            rings         /= np.abs(rings.sum(axis=1))[:, None] # normalization step
            local_output   = local_model.predict(rings)
            # add output
            self.df_[ikey+'_output'] = local_output
            # create the descision
            local_output[ local_output >= thr ] = 1.
            local_output[ local_output < thr ] = 0.
            # add the decision as a new column into the Dataframe
            self.df_[ikey+'_decision'] = local_output
