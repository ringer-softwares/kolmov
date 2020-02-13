__all__ = ['table_info', 'dump_all_train_history', 'dump_train_history']

import os
import re
import glob
import numpy as np
import pandas as pd

import json

# Gaugi dependences
from Gaugi import load as gload


def transform_serialize(d_dict):
    for ikey in d_dict.keys():
        if isinstance(d_dict[ikey], list):
            d_dict[ikey] = np.array(d_dict[ikey], dtype=np.float64).tolist()
    return d_dict

class table_info( object ):
    '''
    The objective of this class is extract the tuning information from saphyra's output and
    create a pandas DataFrame using then.

    The informations used in this DataFrame are listed in info_dict, but the user can add more
    information from saphyra summary for example.
    '''
    def __init__(self, path_to_task_files, config_dict, tag): 
        '''
        Arguments.
        - path_to_task_files: the path to the files that will be used in the extraction

        Ex.: /volume/v1_task/user.mverissi.*/* 
        the example above will get all the tuned files in the v1_task folder

        - config_dict: a dictionary contains in keys the measures that user want to check and
        the values need to be a empty list.

        Ex.: my_info = {
                            # validation
                            'max_sp_val'     : [],
                            'max_sp_pd_val'  : [],
                            'max_sp_fa_val'  : [],
                            'auc_val'        : [],
                            # operation
                            'max_sp_op'     : [],
                            'max_sp_pd_op'  : [],
                            'max_sp_fa_op'  : [],
                            'auc_op'        : [],
                        }

        - tag: a tag for the tuning

        Ex.: v1

        '''
        self.info_dict    = {
            'train_tag'      : [],
            'et_bin'         : [],
            'eta_bin'        : [],
            'model_idx'      : [],
            'sort'           : [],
            'init'           : [],
            # file name
            'file_name'      : [],
            # total
            'total_sgn'      : [],
            'total_bkg'      : [],
        }
        self.tuned_file_list = glob.glob(path_to_task_files)
        self.tag             = tag
        self.config_dict     = config_dict
        self.info_dict.update(self.config_dict)
        self.util_dict = {
            # total
            'total_sgn'     : ('max_sp_pd_op', 2),
            'total_bkg'     : ('max_sp_fa_op', 2),
        }
        print('There are %i files for this task...' %(len(self.tuned_file_list)))


    def get_etbin(self, job):
        return int(re.findall(r'et[a]?[0-9]', job.split('/')[-2])[0][-1])

    def get_etabin(self, job):
        return int(re.findall(r'et[a]?[0-9]', job.split('/')[-2])[1][-1])

    def get_file_name(self, job):
        return job.split('/')[-1]

    def fill_table(self):
        print('Filling the table... ')
        for ituned_file_name in self.tuned_file_list:
            gfile = gload(ituned_file_name)
            tuned_file = gfile['tunedData']
            for ituned in tuned_file:
                summary_dict = ituned['history']['summary']
                # get the basic from model
                self.info_dict['train_tag'].append(self.tag)
                self.info_dict['model_idx'].append(ituned['imodel'])
                self.info_dict['sort'].append(ituned['sort'])
                self.info_dict['init'].append(ituned['init'])
                self.info_dict['et_bin'].append(self.get_etbin(ituned_file_name))
                self.info_dict['eta_bin'].append(self.get_etabin(ituned_file_name))
                self.info_dict['file_name'].append(self.get_file_name(ituned_file_name))
                for t_key in self.util_dict.keys():
                    key, tuple_idx = self.util_dict[t_key]
                    self.info_dict[t_key].append(summary_dict[key][tuple_idx])
                # get the wanted measures
                for imeasure in self.config_dict.keys():
                    if imeasure in list(self.util_dict.keys()):
                        key, tuple_idx = self.util_dict[imeasure] 
                        self.info_dict[imeasure].append(summary_dict[key][tuple_idx])
                    elif isinstance(summary_dict[imeasure], tuple):
                        self.info_dict[imeasure].append(summary_dict[imeasure][0])
                    else:
                        self.info_dict[imeasure].append(summary_dict[imeasure])
                
        self.pandas_table = pd.DataFrame(self.info_dict)
        print('End of fill step, a pandas DataFrame was created...')

    def get_pandas_table(self):
        return self.pandas_table

    def filter_inits(self, key):
        return self.pandas_table.loc[self.pandas_table.groupby(['et_bin', 'eta_bin', 'model_idx', 'sort'])[key].idxmax(), :]
    
    def dump_table(self, cv_table, output_path, table_name):
        cv_table.to_csv(os.path.join(output_path, table_name+'.csv'), index=False)


def dump_train_history(dataframe, 
                        et_bin, eta_bin, 
                        modelidx, sort, path_to_task, output_path):
    # format the task path    
    path_to_task = path_to_task %(et_bin, eta_bin)
    
    # get the job file name and the init for the sort
    print(dataframe.loc[((dataframe.et_bin==eta_bin)     & \
                                        (dataframe.eta_bin==eta_bin)    & \
                                        (dataframe.model_idx==modelidx) & \
                                        (dataframe.sort==sort)          ), ['init', 'file_name']].values[0])
    init, job_name     = dataframe.loc[((dataframe.et_bin==eta_bin)     & \
                                        (dataframe.eta_bin==eta_bin)    & \
                                        (dataframe.model_idx==modelidx) & \
                                        (dataframe.sort==sort)          ), ['init', 'file_name']].values[0]
    # open the choosed file
    tuningfile = gload(os.path.join(path_to_task, job_name))
    h_name = 'history_et%i_eta%i.modelidx_%i.sort_%i' %(et_bin, eta_bin,
                                                        modelidx, sort)
    # extract the information
    for _i in tuningfile['tunedData']:
        if ((_i['imodel'] == modelidx) and \
            (_i['sort'] == sort)       and \
            (_i['init'] == init)):
            with open(os.path.join(output_path, '%s.json' %h_name), 'w') as fp:
                del _i['history']['summary']
                json.dump(transform_serialize(_i['history']), fp)

def dump_all_train_history(dataframe, path_to_task, output_path):

    for iet in dataframe.et_bin.unique():
        for ieta in dataframe.eta_bin.unique():
            for isort in dataframe.sort.unique():
                for imodel in dataframe.model_idx.unique():
                    print('et: %i | eta: %i | sort: %i | model: %i' %(iet, ieta,
                                                                      isort, imodel) )
                    dump_train_history(dataframe,
                                       et_bin=iet, eta_bin=ieta,
                                       sort=isort, modelidx=imodel,
                                       path_to_task=path_to_task, output_path=output_path)
