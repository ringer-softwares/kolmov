
__all__ = ['kplot']

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Gaugi import Logger, expandFolders
from Gaugi.messenger.macros import *

from kolmov.constants import str_etbins_zee, str_etabins

class kplot( Logger ):

    
    def __init__(self, path, model_idx , str_et_bins = str_etbins_zee, 
                                         str_eta_bins = str_etabins
                                         ):
        
        '''
        This class is a little monitoring tool for saphyra trained models.
        Its use the dumped json history of a trained model. If you don't have this json
        use the dump_all_traiplot_monitoringn_history from base_table in kolmov.
        
        Arguments:
        - path_to_history: the path to json history files

        Ex.: /my_volume/my_history_files

        - model_idx: the index of model that you want extract the monitoring info.
        '''

        Logger.__init__( self )
        self.plot_names = {
                            'loss'          : 'Loss Function Evolution',
                            'max_sp_val'    : r'$SP$ Index Evolution',
                            'max_sp_pd_val' : r'$P_D$ Evolution',
                            'max_sp_fa_val' : r'$F_A$ Evolution',
                            }

        self.h_dict         = self.load(path, model_idx)
        self.sort_name_dict =  {idx : 'Fold %i' %(idx+1) for idx in range(10)}
        self.et_range = str_et_bins
        self.eta_range = str_eta_bins



    def load(self, basepath, model_idx):
        
        '''
        This method will open all the histories that was grouped in the initialize method, 
        and put then into a dictionary in order to make easier to manipulate. Usually used
        for histories dumped from best inists table
        '''
        paths = expandFolders( basepath )
        MSG_INFO( self, "Reading %d files...", len(paths) )
        h_dict = dict()
        for path in paths:
            with open(path) as f:
                obj = dict(eval(json.load(f)))
                key = 'et%d_eta%d_sort_%d' %(obj['loc']['et_bin'], obj['loc']['eta_bin'], obj['loc']['sort'])
                if obj['loc']['model_idx'] != model_idx:
                  continue
                h_dict[key] = obj
        return h_dict




    def plot_training_curves(self, et_bin, eta_bin, plot_path, plot_name):
        '''
        Arguments:
        - et_bin: the energy bin index.
        - eta_bin: the pseudorapidity bin index.
        - plot_path: the path to save the plot.
        - plot_name: the plot name.

        Ex.: 
        # initialize the base_ploter class
        plot_tool = plot_monitoring('my_path/awesome_history', model_idx=0)
        # now plot some bin
        plot_tool.plot_monitoring_curves(et_bin=0,
                                         eta_bin=0,
                                         plot_path=my_save_path, 
                                         plot_name=a_very_meaningful_name)
        '''

        # a easy way to lock only in the necessaries keys
        wanted_h_keys = ['et%i_eta%i_sort_%i' %(et_bin, eta_bin, i) for i in range(10)]
        
        
        # train indicators (today we use only this)
        train_indicators = [('loss', 'val_loss'), 'max_sp_val', 'max_sp_pd_val', 'max_sp_fa_val']
        # create a subplots and fill then.
        fig, ax = plt.subplots( nrows=len(wanted_h_keys),
                                ncols=len(train_indicators),
                                figsize=(25,20))
        fig.subplots_adjust(top=0.95, hspace=0.7)
        # loop over the sorts
        for idx, isort in enumerate(wanted_h_keys):
            isort_k = np.int(isort.split('_')[-1])
            # loop over the train indicators
            for jdx, train_ind in enumerate(train_indicators):
                if isinstance(train_ind, tuple):
                    ax[idx, jdx].set_title('%s - %s' %(self.plot_names[train_ind[0]],
                                                       self.sort_name_dict[isort_k]))
                    ax[idx, jdx].set_xlabel('Epochs')
                    ax[idx, jdx].plot(self.h_dict[isort][train_ind[0]], c='r', label='Train Step')
                    ax[idx, jdx].plot(self.h_dict[isort][train_ind[1]], c='b', label='Validation Step')
                    ax[idx, jdx].legend()
                    ax[idx, jdx].grid()
                else:
                    ax[idx, jdx].set_title('%s - %s' %(self.plot_names[train_ind],
                                                       self.sort_name_dict[isort_k]))
                    ax[idx, jdx].set_xlabel('Epochs')
                    ax[idx, jdx].plot(self.h_dict[isort][train_ind], c='g', label='Validation Step')
                    ax[idx, jdx].grid()
        # set the figure name
        fig.suptitle(r'Monitoring Train Plot - $E_T$: %s | $\eta$: %s'
                     %(self.et_range[et_bin], self.eta_range[eta_bin]), fontsize=20)
        # save figure
        fig.savefig(os.path.join(plot_path, '%s.png' %(plot_name)),
                    dpi=300,
                    bbox_inches='tight')


