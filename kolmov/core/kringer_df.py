__all__ = [
    'kringer_df'
]

import os
import json

import numpy as np
import pandas as pd

from tensorflow.keras.models import model_from_json
from Gaugi import Logger
from Gaugi.messenger.macros import *
from Gaugi import load as gload




class kringer_df(Logger):

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
    
    def add_tuning_decision(self, tuned_models_dict):
        '''
        This method will add into the Dataframe the decision from a given tuning.
        
        The tuned_models_dict can be more the one tuning. The key in the dictionary
        will be the key of the decision in the Dataframe.

        Arguments:
        - tuned_models_dict: the dictionary which contains the tunings that you want add 
        the decision.
        Ex.:
        my_tuned_modes = {
            'model_number1' : some-path-to-the-exported-file-with-this-tuning
        }
        '''

        # loop over the items to build a model dict
        for ikey, ituning_file in tuned_models_dict.items():
            MSG_INFO(self, 'Adding %s model to Dataframe' %(ikey))
            # build the model
            aux_file       = gload(ituning_file)
            local_bin      = aux_file['et%i_eta%i' %(self.et_bin, self.eta_bin)]
            local_model    = model_from_json(json.dumps(local_bin['sequential']))
            local_model.set_weights(local_bin['weights'])        
            threshold      = local_bin['threshold']
            # get rings
            rings          = self.get_rings(noHAD=aux_file['noHAD'])
            rings         /= np.abs(rings.sum(axis=1))[:, None] # normalization step
            local_output   = local_model.predict(rings)
            # create the descision
            local_output[ local_output >= threshold ] = 1. 
            local_output[ local_output < threshold ] = 0.
            # add the decision as a new column into the Dataframe
            self.df_[ikey] = local_output