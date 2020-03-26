__all__ = ['table_info', 'dump_all_train_history', 'dump_train_history']

import os
import re
import glob
import numpy as np
import pandas as pd

import json

# Gaugi dependences
from Gaugi import load as gload
import collections
from copy import copy

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
    def __init__(self, path_to_task_files, wanted_keys, tag): 
        '''
        Arguments:
        - path_to_task_files: the path to the files that will be used in the extraction

        Ex.: /volume/v1_task/user.mverissi.*/* 
        the example above will get all the tuned files in the v1_task folder

        - wanted_keys: a dictionary contains in keys the measures that user want to check and
        the values need to be a empty list.

        Ex.: my_info = collections.OrderedDict( {
              
              "max_sp_val"      : 'summary/max_sp_val',
              "max_sp_pd_val"   : 'summary/max_sp_pd_val#0',
              "max_sp_fa_val"   : 'summary/max_sp_fa_val#0',
              "max_sp_op"       : 'summary/max_sp_op',
              "max_sp_pd_op"    : 'summary/max_sp_pd_op#0',
              "max_sp_fa_op"    : 'summary/max_sp_fa_op#0',
              'tight_pd_ref'    : "reference/tight_cutbased/pd_ref#0",
              'tight_fa_ref'    : "reference/tight_cutbased/fa_ref#0",
              } )

        - tag: a tag for the tuning

        Ex.: v1
        '''

        self.tag             = tag
        self.tuned_file_list = glob.glob(path_to_task_files)
        # Check wanted key type 
        if type(wanted_keys) is not collections.OrderedDict:
          TypeError("The wanted key must be an collection.OrderedDict to preserve the order inside of the dataframe.")
        self.wanted_keys     = wanted_keys

        self.dataframe    = collections.OrderedDict({
                              'train_tag'      : [],
                              'et_bin'         : [],
                              'eta_bin'        : [],
                              'model_idx'      : [],
                              'sort'           : [],
                              'init'           : [],
                              'file_name'      : [],
                          })


        self.wanted_keys.update(
          {
            'total_sgn'     : 'summary/max_sp_pd_op#2',
            'total_bkg'     : 'summary/max_sp_fa_op#2',
          }
        )
        
        self.configure()
        print('There are %i files for this task...' %(len(self.tuned_file_list)))

    
    def configure(self):
      '''
      Configure all keys into the dataframe.
      '''
      for key in self.wanted_keys:
        # varname:summary/var1#2
        varname = key.split(':')[0] 
        # Add varname to the dataframe
        self.dataframe[varname] = [] 


    def get_etbin(self, job):
        '''
        A simple method to get the et bin in for a task.
        Arguments:
        -job: A item of tuned_file_list
        '''
        return int(re.findall(r'et[a]?[0-9]', job.split('/')[-2])[0][-1])

    def get_etabin(self, job):
        '''
        A simple method to get the eta bin in for a task.
        Arguments:
        -job: A item of tuned_file_list
        '''
        return int(re.findall(r'et[a]?[0-9]', job.split('/')[-2])[1][-1])

    def get_file_name(self, job):
        '''
        A simple method to get the name of the job file.
        Arguments:
        -job: A item of tuned_file_list
        '''
        return job.split('/')[-1]

    def fill_table(self):
        '''
        This method will fill the information dictionary and convert then into a pandas DataFrame.
        '''
        print('Filling the table... ')
        for ituned_file_name in self.tuned_file_list:
            gfile = gload(ituned_file_name)
            tuned_file = gfile['tunedData']
            for ituned in tuned_file:
                history = ituned['history']
                # get the basic from model
                self.dataframe['train_tag'].append(self.tag)
                self.dataframe['model_idx'].append(ituned['imodel'])
                self.dataframe['sort'].append(ituned['sort'])
                self.dataframe['init'].append(ituned['init'])
                self.dataframe['et_bin'].append(self.get_etbin(ituned_file_name))
                self.dataframe['eta_bin'].append(self.get_etabin(ituned_file_name))
                self.dataframe['file_name'].append(self.get_file_name(ituned_file_name))
                # Get the value for each wanted key passed by the user in the contructor args.
                for key, local  in self.wanted_keys.items():
                  self.dataframe[key].append( self.get_value( history, local ) )

        self.pandas_table = pd.DataFrame(self.dataframe)
        print('End of fill step, a pandas DataFrame was created...')


    def get_value(self, history, local):
        '''
        Return the value using recursive navigation
        '''
        # Protection to not override the history since this is a 'mutable' object
        var = copy(history)
        for key in local.split('/'):
            var = var[key.split('#')[0]][int(key.split('#')[1])] if '#' in key else var[key]
        return var
    
    def get_pandas_table(self):
        '''
        Return the pandas table created in fill_table
        '''
        return self.pandas_table


    def filter_inits(self, key):
        '''
        This method will filter the pandas DataFrame in order to get the best initialization for
        each sort. This filter need a key to ordenate the initialization and filter.
        Arguments:
        -key: a measure to use for filter the initializations

        Ex.: 
        # create a table_info
        Example = table_info(my_task_full_name, my_tuned_info, tag='my_flag')
        # fill table
        Example.fill_table()
        # now filter the inits using a max_sp_val
        my_df = Example.filter_inits(key='max_sp_val')

        In this example my_df is a pandas DataFrame with the best intilization sorted by 'max_sp_val'.

        '''
        return self.pandas_table.loc[self.pandas_table.groupby(['et_bin', 'eta_bin', 'model_idx', 'sort'])[key].idxmax(), :]
   


    def calculate_mean_and_std(self, cv_table, wanted_keys):
        '''
        Calculate the mean and std for wanted keys with val and op suffix.
        This functions is usefull to check the fluctuation for some keys for
        all sorts inside a bin/tuning.
        '''
        print( "Calculate mean and std values for this table.")
        # Create a new dataframe to hold this table
        dataframe = { 'train_tag' : [], 'et_bin' : [], 'eta_bin' : []}
        # Include all wanted keys into the dataframe
        for key in wanted_keys:
          if ('op' in key) or ('val' in key):
            # Insert mean and std column for val and op keys
            dataframe[key+'_mean'] = []
            dataframe[key+'_std'] = []
          else:
            dataframe[key] = []

        # Loop over all tuning tags and et/eta bins
        for tag in cv_table['train_tag'].unique():
            for et_bin in cv_table['et_bin'].unique():
                for eta_bin in cv_table['eta_bin'].unique():
                  # Filter by tag, et_bin and eta_bin
                  cv_bin = cv_table.loc[ (cv_table['train_tag'] == tag) & (cv_table['et_bin'] == et_bin) & (cv_table['eta_bin'] == eta_bin) ]
                  dataframe['train_tag'].append( tag )
                  dataframe['et_bin'].append( et_bin )
                  dataframe['eta_bin'].append( eta_bin )
                  for key in wanted_keys:
                      if ('op' in key) or ('val' in key):
                          dataframe[key+'_mean'].append( cv_bin[key].mean() )
                          dataframe[key+'_std'].append( cv_bin[key].std() )
                      else:
                          dataframe[key].append( cv_bin[key].unique()[0] )
        # Return the pandas dataframe
        return pd.DataFrame(dataframe)




    def dump_table(self, cv_table, output_path, table_name):
        '''
        A method to save the pandas DataFrame created by table_info into a .csv file.
        Arguments:
        - cv_table: the table created (and filtered) by table_info;
        - output_path: the file destination;
        - table_name: the file name;

        Ex.:
        # create a table_info
        Example = table_info(my_task_full_name, my_tuned_info, tag='my_flag')
        # fill table
        Example.fill_table()
        # now filter the inits using a max_sp_val
        my_df = Example.filter_inits(key='max_sp_val')
        # now save my_df
        Example.dump_table(my_df, my_path, 'a_very_meaningful_name')

        In this example, a file containing my_df, called 'a_very_meaningful_name.csv', will be saved in my_path
        '''
        cv_table.to_csv(os.path.join(output_path, table_name+'.csv'), index=False)



def dump_train_history(dataframe, 
                        et_bin, eta_bin, 
                        modelidx, sort, path_to_task, output_path):
    '''
    This function get the job file search for the right model, init and sort and dump the
    training history into a json file.
    Arguments:
    - dataframe: a pandas DataFrame which contains the tuning information;
    - et_bin: the et bin index;
    - eta_bin: the eta bin index;
    - modelidx: the model index;
    - sort: the sort index;
    - path_to_task: the path to the job files;
    - output_path: the destination path.
    '''
    # format the task path    
    path_to_task = path_to_task %(et_bin, eta_bin)
    
    # get the job file name and the init for the sort
    init, job_name     = dataframe.loc[((dataframe.et_bin==et_bin)      & \
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
    '''
    This function get all history for each (et, eta) bin and save into a json file.
    Arguments:
    - dataframe: a pandas DataFrame which contains the tuning information;
    - path_to_task: the path to the job files;
    - output_path: the destination path.
    '''
    for iet in dataframe.et_bin.unique():
        for ieta in dataframe.eta_bin.unique():
            for isort in dataframe.sort.unique():
                for imodel in dataframe.model_idx.unique():
                    dump_train_history(dataframe,
                                       et_bin=iet, eta_bin=ieta,
                                       sort=isort, modelidx=imodel,
                                       path_to_task=path_to_task, output_path=output_path)
