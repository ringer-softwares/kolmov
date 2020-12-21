__all__ = [
    'first_export_tool',
    'export_tool',
    'export_onnx_tool',
    'export_fastnet_to_onnx'
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

from kolmov.utils.constants import etbins_zee, etbins_jpsiee, etabins

import onnx
import keras2onnx
from ROOT import TEnv, TFile

# to get the legacy correctly
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


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

# to export in onnx format

class export_fastnet_to_onnx(object):
    '''
    This class was created to export tunings that were made in TuningTools (FastNet) in order to move them to prometheus framework.
    '''

    def __init__(self, path_to_tuning):
        '''
        Arguments:
        path_to_tunings: the path that contains the .root files
        '''
        self.thr_filename    = os.path.join(path_to_tuning, 'TrigL2CaloRingerElectron%sThresholds.root')
        self.models_filename = os.path.join(path_to_tuning, 'TrigL2CaloRingerElectron%sConstants.root')

    # auxiliar function to create the export files
    def list_to_str(self, l ):
        s = str()
        for ll in l:
            s+=str(ll)+'; '
        return s[:-2]

    def translate_bins(self, et, eta, isJpsiee=True):
        '''
        This function will translate the bins from root file.
        Arguments:
        et: et value from root file
        eta: eta value from root file
        isJpsiee: boolean to specific the et bins.
        '''
        if isJpsiee:
            if et[0] == 0:
                etBin = 0
            elif et[0] == 7:
                etBin = 1
            elif et[0] == 10:
                etBin = 2
        else:
            if et[0] == 0 or et[0] == 15:
                etBin = 0
            elif et[0] == 20:
                etBin = 1
            elif et[0] == 30:
                etBin = 2
            elif et[0] == 40:
                etBin = 3
            elif et[0] == 50:
                etBin = 4
        # get eta
        if eta[0] == 0:
            etaBin = 0
        elif eta[0] == 0.8:
            etaBin = 1
        elif eta[0] == 1.37:
            etaBin = 2
        elif eta[0] == 1.54:
            etaBin = 3
        elif eta[0] == 2.37:
            etaBin = 4
        return etBin, etaBin

    def convert_to_keras_model(self, neuron, W, B, input_shape=100 ):
        '''
        This function will create a keras model which will be filled with FastNet model information.
        This function works only and only for MLP models with one hidden layer.
        Arguments:
        neuron: #neurons in hidden layer.
        W: FastNet weights.
        B: FastNet bias.
        input_shape: shape of input layer
        '''
        # Build the standard model
        model  = Sequential()
        model.add( Dense( neuron , input_shape= (input_shape,), activation = 'tanh') )
        model.add( Dense( 1, activation='linear' ) )
        w0 = np.zeros((neuron, input_shape))
        b0 = []
        for i in range(neuron):
            for j in range(input_shape):
                w0[i][j] = W.pop(0)
            b0.append(B.pop(0))
        w1 = W
        b1 = B
        ww = np.array([ np.array(w0).T, np.array(b0), np.array(w1), np.array(b1) ] )
        ww[2]=ww[2].reshape((ww[2].shape[0],1))
        model.set_weights(ww)
        return model

    def create_config_files(self, operation_point, tuning_name, version,
                            model_format_name, output_config_name,
                            signature='electron', isJpsiee=True):
        '''
        This function will fill the dictionary of operation models using the information
        from the operation dataframe and using the ringer data to calculate the threshold.
        '''

        etabin_list = etabins
        if isJpsiee:
            etbin_list = etbins_jpsiee
        else:
            etbin_list = etbins_zee

        # format the file names, open then and get the tree
        thrs_file      = TFile(self.thr_filename %(operation_point))
        thresholdsTree = thrs_file.tuning.Get('thresholds')

        models_file    = TFile(self.models_filename %(operation_point))
        modelsTree     = models_file.tuning.Get('discriminators')

        # tuning information
        model_etmin_vec = []
        model_etmax_vec = []
        model_etamin_vec = []
        model_etamax_vec = []
        model_paths = []

        # slopes and offsets (threshold configuration)
        slopes  = []
        offsets = []
        # loop over models and thresholds in TTree
        for ithr, imodel in zip(thresholdsTree, modelsTree):
            # get the bins
            iet, ieta = self.translate_bins(ithr.etBin, ithr.etaBin)
            print('Processing et bin: %i | eta bin: %i' %(iet, ieta))

            # fill et and eta vec
            model_etmin_vec.append(etbin_list[iet][0])
            model_etmax_vec.append(etbin_list[iet][1])
            model_etamin_vec.append(etabin_list[ieta][0])
            model_etamax_vec.append(etabin_list[ieta][1])

            # easy way to get the thresholds
            l_thr   = list(ithr.thresholds)
            slopes.append(l_thr[0])
            offsets.append(l_thr[1])

            # get the models
            weights = list(imodel.weights)
            bias = list(imodel.bias)
            l_model = self.convert_to_keras_model(len(weights)%100, weights, bias)
            print(l_model.summary())
            # convert keras to Onnx
            onnx_model = keras2onnx.convert_keras(l_model, l_model.name)
            # onnx model name and add to the model paths
            s_l = model_format_name %(operation_point, iet, ieta)
            # now add the model string to model_paths
            onnx_model_name = s_l
            model_paths.append( s_l.split('/')[-1] )
            # save the onnx model
            onnx.save_model(onnx_model, onnx_model_name)
        # done! Fill all list with the models
        # write the config file
        m_file = TEnv('ringer')
        m_file.SetValue("__name__", tuning_name)
        m_file.SetValue("__version__", version)
        m_file.SetValue("__operation__", operation_point)
        m_file.SetValue("__signature__", signature)
        m_file.SetValue("Model__size", str(len(model_paths)))
        m_file.SetValue("Model__etmin", self.list_to_str(model_etmin_vec))
        m_file.SetValue("Model__etmax", self.list_to_str(model_etmax_vec))
        m_file.SetValue("Model__etamin", self.list_to_str(model_etamin_vec))
        m_file.SetValue("Model__etamax", self.list_to_str(model_etamax_vec))
        m_file.SetValue("Model__path", self.list_to_str(model_paths))
        m_file.SetValue( "Threshold__size"  , str(len(model_paths)) )
        m_file.SetValue( "Threshold__etmin" , self.list_to_str(model_etmin_vec) )
        m_file.SetValue( "Threshold__etmax" , self.list_to_str(model_etmax_vec) )
        m_file.SetValue( "Threshold__etamin", self.list_to_str(model_etamin_vec) )
        m_file.SetValue( "Threshold__etamax", self.list_to_str(model_etamax_vec) )
        m_file.SetValue( "Threshold__slope" , self.list_to_str(slopes) )
        m_file.SetValue( "Threshold__offset", self.list_to_str(offsets) )
        m_file.SetValue( "Threshold__MaxAverageMu", 100.)
        m_file.WriteFile(output_config_name%(operation_point))



class export_onnx_tool(object):
    '''
    This class is the default kolmov export tool to get and prepare the tunings
    to move them for prometheus framework in onnx format.
    '''

    def __init__(self, operation_dataframe):
        '''
        Arguments:
        - operation_dataframe: a .csv file with all models selected to operation.
        '''
        self.op_df            = pd.read_csv(operation_dataframe)


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

    # auxiliar function to create the export files
    def list_to_str(self, l ):
        s = str()
        for ll in l:
            s+=str(ll)+'; '
        return s[:-2]

    def create_config_files(self, operation_point, tuning_name, version,
                            model_format_name, output_config_name,
                            signature='electron', isJpsiee=True):
        '''
        This function will fill the dictionary of operation models using the information
        from the operation dataframe and using the ringer data to calculate the threshold.
        '''

        etabin_list = etabins
        if isJpsiee:
            etbin_list = etbins_jpsiee
        else:
            etbin_list = etbins_zee

        thr_dataframe_key = aux_threshold_dict_keys[operation_point]
        # loop over et and eta
        # tuning information
        model_etmin_vec = []
        model_etmax_vec = []
        model_etamin_vec = []
        model_etamax_vec = []
        model_paths = []
        # slopes and offsets (threshold configuration)
        slopes  = []
        offsets = []
        # loop over et and eta
        for iet, ieta in product(range(self.op_df.et_bin.nunique()),
                                 range(self.op_df.eta_bin.nunique())):
            print('Processing et bin: %i | eta bin: %i' %(iet, ieta))


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

            # fill et and eta vec
            model_etmin_vec.append(etbin_list[iet][0])
            model_etmax_vec.append(etbin_list[iet][1])
            model_etamin_vec.append(etabin_list[ieta][0])
            model_etamax_vec.append(etabin_list[ieta][1])

            # set the weights to the model
            l_model = model_from_json(json.dumps(m))
            l_model.set_weights(w)
            # remove the tanh in output
            l_model.pop()
            print(l_model.summary())
            # convert keras to Onnx
            onnx_model = keras2onnx.convert_keras(l_model, l_model.name)
            # onnx model name and add to the model paths
            s_l = model_format_name %(operation_point, iet, ieta)

            onnx_model_name = s_l
            model_paths.append( s_l.split('/')[-1] )
            # save the onnx model
            onnx.save_model(onnx_model, onnx_model_name)

            # add dummy thresholds
            slopes.append(0.)
            offsets.append(0.)
        # done! Fill all list with the models
        # write the config file
        m_file = TEnv('ringer')
        m_file.SetValue("__name__", tuning_name)
        m_file.SetValue("__version__", version)
        m_file.SetValue("__operation__", operation_point)
        m_file.SetValue("__signature__", signature)
        m_file.SetValue("Model__size", str(len(model_paths)))
        m_file.SetValue("Model__etmin", self.list_to_str(model_etmin_vec))
        m_file.SetValue("Model__etmax", self.list_to_str(model_etmax_vec))
        m_file.SetValue("Model__etamin", self.list_to_str(model_etamin_vec))
        m_file.SetValue("Model__etamax", self.list_to_str(model_etamax_vec))
        m_file.SetValue("Model__path", self.list_to_str(model_paths))
        m_file.SetValue( "Threshold__size"  , str(len(model_paths)) )
        m_file.SetValue( "Threshold__etmin" , self.list_to_str(model_etmin_vec) )
        m_file.SetValue( "Threshold__etmax" , self.list_to_str(model_etmax_vec) )
        m_file.SetValue( "Threshold__etamin", self.list_to_str(model_etamin_vec) )
        m_file.SetValue( "Threshold__etamax", self.list_to_str(model_etamax_vec) )
        m_file.SetValue( "Threshold__slope" , self.list_to_str(slopes) )
        m_file.SetValue( "Threshold__offset", self.list_to_str(offsets) )
        m_file.SetValue( "Threshold__MaxAverageMu", 100.)
        m_file.WriteFile(output_config_name%(operation_point))

########################
#TOGUIDE
# def convert_to_onnx_with_dummy_thresholds( models, name, version, signature, model_output_format , operation, output):

#     import onnx
#     import keras2onnx
#     from ROOT import TEnv
#     model_etmin_vec = []
#     model_etmax_vec = []
#     model_etamin_vec = []
#     model_etamax_vec = []
#     model_paths = []

#     slopes = []
#     offsets = []

#     for model in models:

#         model_etmin_vec.append( model['etBin'][0] )
#         model_etmax_vec.append( model['etBin'][1] )
#         model_etamin_vec.append( model['etaBin'][0] )
#         model_etamax_vec.append( model['etaBin'][1] )

#         etBinIdx = model['etBinIdx']
#         etaBinIdx = model['etaBinIdx']

#         # Conver keras to Onnx
#         onnx_model = keras2onnx.convert_keras(model['model'], model['model'].name)

#         onnx_model_name = model_output_format%( etBinIdx, etaBinIdx )
#         model_paths.append( onnx_model_name )

#         # Save onnx mode!
#         onnx.save_model(onnx_model, onnx_model_name)

#         slopes.append( 0.0 )
#         offsets.append( 0.0 )


#     def list_to_str( l ):
#         s = str()
#         for ll in l:
#           s+=str(ll)+'; '
#         return s[:-2]

#     # Write the config file
#     file = TEnv( 'ringer' )
#     file.SetValue( "__name__", name )
#     file.SetValue( "__version__", version )
#     file.SetValue( "__operation__", operation )
#     file.SetValue( "__signature__", signature )
#     file.SetValue( "Model__size"  , str(len(models)) )
#     file.SetValue( "Model__etmin" , list_to_str(model_etmin_vec) )
#     file.SetValue( "Model__etmax" , list_to_str(model_etmax_vec) )
#     file.SetValue( "Model__etamin", list_to_str(model_etamin_vec) )
#     file.SetValue( "Model__etamax", list_to_str(model_etamax_vec) )
#     file.SetValue( "Model__path"  , list_to_str( model_paths ) )
#     file.SetValue( "Threshold__size"  , str(len(models)) )
#     file.SetValue( "Threshold__etmin" , list_to_str(model_etmin_vec) )
#     file.SetValue( "Threshold__etmax" , list_to_str(model_etmax_vec) )
#     file.SetValue( "Threshold__etamin", list_to_str(model_etamin_vec) )
#     file.SetValue( "Threshold__etamax", list_to_str(model_etamax_vec) )
#     file.SetValue( "Threshold__slope" , list_to_str(slopes) )
#     file.SetValue( "Threshold__offset", list_to_str(offsets) )
#     file.SetValue( "Threshold__MaxAverageMu", 100)
#     file.WriteFile(output)

# for op in ['Tight','Medium','Loose','VeryLoose']:
#     format = 'data17_13TeV_EGAM1_probes_lhmedium_EGAM7_vetolhvloose.model_v10.electron'+op+'.et%d_eta%d.onnx'
#     output = "ElectronRinger%sTriggerConfig.conf"%op
#     convert_to_onnx_with_dummy_thresholds( models, 'TrigL2_20200715_v10', 'v10', 'electron', format ,op ,output)

################

# this tool will calculate the threshold in case that this info aren't avaliable
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



