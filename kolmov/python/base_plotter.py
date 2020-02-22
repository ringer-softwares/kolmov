__all__ = ['plot_monitoring']

import os
import glob
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

class plot_monitoring(object):
    '''
    This class is a little monitoring tool for saphyra trained models.

    Its use the dumped json history of a trained model. If you don't have this json
    use the dump_all_train_history from base_table in kolmov.
    '''
    def __init__(self, path_to_history, model_idx, low_et=True):
        '''
        Arguments:
        - path_to_history: the path to json history files

        Ex.: /my_volume/my_history_files

        - model_idx: the index of model that you want extract the monitoring info.

        Ex.: model_idx = 0 
        In this example 0 is the first model from a list of tested model.

        - low_et: a flag to use Jpsi eT kinematic region or Zee. Default is True and
        Jpsi region is used. If you need use the Zee kinematic region set this flag to False.
        '''
        self.list_h_files = (glob
                            .glob(os
                            .path
                            .join(path_to_history, '*.modelidx_%i*' %(model_idx))))
        self.plot_names = {
            'loss'          : 'Loss Function Evolution',
            'max_sp_val'    : r'$SP$ Index Evolution',
            'max_sp_pd_val' : r'$P_D$ Evolution',
            'max_sp_fa_val' : r'$F_A$ Evolution',
            }
        self.h_dict         = self.open_history()
        self.sort_name_dict =  {idx : 'Fold %i' %(idx+1) for idx in range(10)}
        # set the real ranges in kinematic regions 
        # check if is Zee eT bins or Jpsi.
        if not low_et: #Zee
            self.et_range = ['[15, 20[ GeV',
                            '[20, 30[ GeV',
                            '[30, 40[ GeV',
                            '[40, 50[ GeV',
                            r'[50, $\infty$[']
        else: # Jpsi - now is the default
            self.et_range = ['[4, 7[ GeV',
                            '[7, 10[ GeV',
                            '[10, 15[ GeV']
        self.eta_range = ['[0.0, 0.8[',
                          '[0.8, 1.37[',
                          '[1.37, 1.54[',
                          '[1.54, 2.37[',
                          '[2.37, 2.47[']

    def open_history(self):
        '''
        This method will open all the histories that was grouped in the initialize method, 
        and put then into a dictionary in order to make easier to manipulate.
        '''
        h_dict = {}
        for h_file in self.list_h_files:
            f_split         = h_file.split('.')
            et_bin, eta_bin = f_split[0].split('_')[-2], f_split[0].split('_')[-1]
            sort_           = f_split[-2]
            key             = '%s_%s_%s' %(et_bin, eta_bin, sort_)dump_all_train_history
            with open(h_file) as jfile:
                h_dict[key] = json.load(jfile)

        return h_dict


    def plot_monitoring_curves(self, et_bin, eta_bin, plot_path, plot_name):
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
                                        plot_path=my_save_path, plot_name=a_very_meaningful_name)
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