__all__ = [
    'calc_sp',
    'first_export_tool',
    'export_tool'
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

from kolmov.core.constants import etbins_zee, etbins_jpsiee, etabins
# SP index definition
def calc_sp(pd, fa):
    g_mean = np.sqrt((1-fa) * pd)
    a_mean = ((1-fa) + pd)/2.
    return np.sqrt(g_mean*a_mean)

aux_threshold_dict_keys = {
    'Tight'     : 'tight_op_threshold',
    'Medium'    : 'medium_op_threshold',
    'Loose'     : 'loose_op_threshold',
    'VeryLoose' : 'vloose_op_threshold',
}

# this export has the threshold info, so don't need to preed    
class export_tool(object):
    '''
    This class is the default kolmov export tool to get and prepare the tunings
    to move them for prometheus framework. 
    '''

    def __init__(self, operation_dataframe):
        '''
        Arguments:
        - operation_dataframe: a .csv file with all models selected to operation.
        '''
        self.op_df            = pd.read_csv(operation_dataframe)
    

    def get_models_dict(self):
        return self.model_dict

    def get_threshold_dict(self):
        return self.threshold_dict

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
    
    def save_dicts(self, operation_point):
        '''
        This function will save a dictionary into a json file.

        Arguments:
        -dictionary: is the python dict that you want to save;
        -filename: the name of dictionary to be save.
        '''
        thr_filename = 'TrigL2CaloRingerElectron%sThresholds.json' %operation_point
        m_filename   = 'TrigL2CaloRingerElectron%sConstants.json' %operation_point
        with open(thr_filename, 'w') as fp:
            json.dump(self.threshold_dict, fp)
        with open(m_filename, 'w') as fp:
            json.dump(self.model_dict, fp)


    def fill_models_thr_dict(self, operation_point, tuning_tag,
                             tuning_name, isJpsiee=True, save_json=True):
        '''
        This function will fill the dictionary of operation models using the information
        from the operation dataframe and using the ringer data to calculate the threshold.
        '''

        # models dictionary
        self.model_dict = {}
        self.model_dict['models']          = [] # this must be a list
        self.model_dict['__version__']     = tuning_tag
        self.model_dict['__type__']        = 'Model'
        self.model_dict['__name__']        = tuning_name
        self.model_dict['__description__'] = ''
        # threshold dictionary
        self.threshold_dict = {}
        self.threshold_dict['thresholds']      = [] # same as model
        self.threshold_dict['__version__']     = tuning_tag
        self.threshold_dict['__type__']        = 'Threshold'
        self.threshold_dict['__name__']        = tuning_name
        self.threshold_dict['__description__'] = ''

        etabin_list = etabins
        if isJpsiee:
            etbin_list = etbins_jpsiee
        else:
            etbin_list = etbins_zee
        
        # set the operation label
        self.model_dict['__operation__'] = operation_point
        self.threshold_dict['__operation__'] = operation_point
        thr_dataframe_key = aux_threshold_dict_keys[operation_point]        

        for iet, ieta in product(range(self.op_df.et_bin.nunique()),
                                 range(self.op_df.eta_bin.nunique())):
            print('Processing et bin: %i | eta bin: %i' %(iet, ieta))
            
            # we need a local dictionary
            m_local_dict = {}
            thr_local_dict = {}
            # 1 step: get the right file, model_id, sort and init to open.
            aux_df = self.op_df.loc[((self.op_df['et_bin'] == iet) &\
                                     (self.op_df['eta_bin'] == ieta)), :].copy()
            f_name, id_model, sort, init, thr = aux_df.iloc[0][['file_name', \
                                                                'model_idx', \
                                                                'sort', \
                                                                'init', \
                                                                thr_dataframe_key]].values
            # model info: sequential, weights and binning information
            # add the threshold to thr_local dictionary
            m, w = self.model_finder(tunedfile=gload(f_name),
                                     model_idx=id_model,
                                     sort=sort,
                                     init=init)
            m_local_dict['sequential'] = m
            m_local_dict['weights']    = [wi.tolist() for wi in w] #np.arrays are not serialized
            m_local_dict['etBin']  = etbin_list[iet]
            m_local_dict['etaBin'] = etabin_list[ieta]
            
            # thr info: threshold and binning information
            # this information must to be a list with 3 itens [alpha, beta, raw_threshold]
            # since wee do not have pile up correction in saphyra the configuration must to be
            # [0., raw_threshold, raw_threshold]
            thr_local_dict['threshold'] = [0., thr, thr]
            thr_local_dict['etBin']  = etbin_list[iet]
            thr_local_dict['etaBin'] = etabin_list[ieta]

            # append the local dict to dict
            self.model_dict['models'].append(m_local_dict)
            self.threshold_dict['thresholds'].append(thr_local_dict)
        if save_json:
            print('Saving json files...')
            self.save_dicts(operation_point)
        print('Done!')

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


    
