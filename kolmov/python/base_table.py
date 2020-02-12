__all__ = ['table_info']

import os
import re
import glob
import numpy as np
import pandas as pd

# Gaugi dependences
from Gaugi import load as gload

class table_info( object ):
    
    def __init__(self, path_to_task_files, config_dict, tag): 
        
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
