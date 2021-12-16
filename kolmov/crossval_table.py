

__all__ = ['crossval_table']

from Gaugi.macros import *
from Gaugi import Logger, expand_folders, load, progressbar
from pybeamer import *
from pprint import pprint
from functools import reduce
from itertools import product
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import collections, os, glob, json, copy, re
import numpy as np
import pandas as pd
import os
import json
import joblib
import tensorflow as tf
model_from_json = tf.keras.models.model_from_json



class crossval_table( Logger ):
    #
    # Constructor
    #
    def __init__(self, config_dict, etbins=None, etabins=None ):
        '''
        The objective of this class is extract the tuning information from saphyra's output and
        create a pandas DataFrame using then.
        The informations used in this DataFrame are listed in info_dict, but the user can add more
        information from saphyra summary for example.


        Arguments:

        - config_dict: a dictionary contains in keys the measures that user want to check and
        the values need to be a empty list.

        Ex.: info = collections.OrderedDict( {

              "max_sp_val"      : 'summary/max_sp_val',
              "max_sp_pd_val"   : 'summary/max_sp_pd_val#0',
              "max_sp_fa_val"   : 'summary/max_sp_fa_val#0',
              "max_sp_op"       : 'summary/max_sp_op',
              "max_sp_pd_op"    : 'summary/max_sp_pd_op#0',
              "max_sp_fa_op"    : 'summary/max_sp_fa_op#0',
              'tight_pd_ref'    : "reference/tight_cutbased/pd_ref#0",
              'tight_fa_ref'    : "reference/tight_cutbased/fa_ref#0",
              'tight_pd_ref_passed'     : "reference/tight_cutbased/pd_ref#1",
              'tight_fa_ref_passed'     : "reference/tight_cutbased/fa_ref#1",
              'tight_pd_ref_total'      : "reference/tight_cutbased/pd_ref#2",
              'tight_fa_ref_total'      : "reference/tight_cutbased/fa_ref#2",
              'tight_pd_val_passed'     : "reference/tight_cutbased/pd_val#1",
              'tight_fa_val_passed'     : "reference/tight_cutbased/fa_val#1",
              'tight_pd_val_total'      : "reference/tight_cutbased/pd_val#2",
              'tight_fa_val_total'      : "reference/tight_cutbased/fa_val#2",
              'tight_pd_op_passed'      : "reference/tight_cutbased/pd_op#1",
              'tight_fa_op_passed'      : "reference/tight_cutbased/fa_op#1",
              'tight_pd_op_total'       : "reference/tight_cutbased/pd_op#2",
              'tight_fa_op_total'       : "reference/tight_cutbased/fa_op#2",

              } )

        - etbins: a list of et bins edges used in training;
        - etabins: a list of eta bins edges used in training;
        '''
        Logger.__init__(self)
        self.__table = None
        # Check wanted key type
        self.__config_dict = collections.OrderedDict(config_dict) if type(config_dict) is dict else config_dict
        self.__etbins = etbins
        self.__etabins = etabins


    #
    # Fill the main dataframe with values from the tuning files and convert to pandas dataframe
    #
    def fill(self, path, tag):
        '''
        This method will fill the information dictionary and convert then into a pandas DataFrame.

        Arguments.:

        - path: the path to the tuned files;
        - tag: the training tag used;
        '''
        paths = expand_folders( path )
        MSG_INFO(self, "Reading file for %s tag from %s", tag , path)

        # Creating the dataframe
        dataframe = collections.OrderedDict({
                              'train_tag'      : [],
                              'et_bin'         : [],
                              'eta_bin'        : [],
                              'model_idx'      : [],
                              'sort'           : [],
                              'init'           : [],
                              'file_name'      : [],
                              'tuned_idx'      : [],
                              'op_name'        : [],
                          })

        MSG_INFO(self, 'There are %i files for this task...' %(len(paths)))
        MSG_INFO(self, 'Filling the table... ')

        for ituned_file_name in progressbar( paths , 'Reading %s...'%tag):
            #for ituned_file_name in paths:

            try:
                gfile = load(ituned_file_name)
            except:
                #MSG_WARNING(self, "File %s not open. skip.", ituned_file_name)
                continue
            tuned_file = gfile['tunedData']

            for idx, ituned in enumerate(tuned_file):

                history = ituned['history']
                
                for op, config_dict in self.__config_dict.items():

                    # get the basic from model
                    dataframe['train_tag'].append(tag)
                    dataframe['model_idx'].append(ituned['imodel'])
                    dataframe['sort'].append(ituned['sort'])
                    dataframe['init'].append(ituned['init'])
                    dataframe['et_bin'].append(self.get_etbin(ituned_file_name))
                    dataframe['eta_bin'].append(self.get_etabin(ituned_file_name))
                    dataframe['file_name'].append(ituned_file_name)
                    dataframe['tuned_idx'].append( idx )
                    dataframe['op_name'].append(op)

                    # Get the value for each wanted key passed by the user in the contructor args.
                    for key, local  in config_dict.items():
                        if not key in dataframe.keys():
                            dataframe[key] = [self.__get_value( history, local )]
                        else:
                            dataframe[key].append( self.__get_value( history, local ) )

        # append tables if is need
        # ignoring index to avoid duplicated entries in dataframe
        self.__table = self.__table.append( pd.DataFrame(dataframe), ignore_index=True ) if not self.__table is None else pd.DataFrame(dataframe)
        MSG_INFO(self, 'End of fill step, a pandas DataFrame was created...')


    #
    # Convert the table to csv
    #
    def to_csv( self, output ):
        '''
        This function will save the pandas Dataframe into a csv file.

        Arguments.:

        - output: the path and the name to be use for save the table.

        Ex.:
        m_path = './my_awsome_path
        m_name = 'my_awsome_name.csv'

        output = os.path.join(m_path, m_name)
        '''
        self.__table.to_csv(output, index=False)


    #
    # Read the table from csv
    #
    def from_csv( self, input ):
        '''
        This method is used to read a csv file insted to fill the Dataframe from tuned file.

        Arguments:

        - input: the csv file to be opened;
        '''
        self.__table = pd.read_csv(input)


    #
    # Get the main table
    #
    def table(self):
        '''
        This method will return the pandas Dataframe.
        '''
        return self.__table


    #
    # Set the main table
    #
    def set_table(self, table):
        '''
        This method will set a table indo the main class table.

        Arguments:

        - table: a pandas Dataframe;
        '''
        self.__table=table


    #
    # Get the value using recursive dictionary navigation
    #
    def __get_value(self, history, local):
        '''
        This method will return a value given a history and dictionary with keys.

        Arguments:

        - history: the tuned information file;
        - local: the path caming from config_dict;
        '''
        # Protection to not override the history since this is a 'mutable' object
        var = copy.copy(history)
        for key in local.split('/'):
            var = var[key.split('#')[0]][int(key.split('#')[1])] if '#' in key else var[key]
        return var


    def get_etbin(self, job):
        return int(  re.findall(r'et[a]?[0-9]', job)[0][-1] )


    def get_etabin(self, job):
        return int( re.findall(r'et[a]?[0-9]',job)[1] [-1] )


    def get_etbin_edges(self, et_bin):
        return (self.__etbins[et_bin], self.__etbins[et_bin+1])


    def get_etabin_edges(self, eta_bin):
        return (self.__etabins[eta_bin], self.__etabins[eta_bin+1])


    #
    # Get the pandas dataframe
    #
    def table(self):
        '''
        This method will return the pandas Dataframe.
        '''
        return self.__table


    #
    # Return only best inits
    #
    def filter_inits(self, key, idxmin=False):
        '''
        This method will filter the Dataframe based on given key in order to get the best inits for every sort.

        Arguments:

        - key: the column to be used for filter.
        '''
        if idxmin:
            idxmask = self.table().groupby(['et_bin', 'eta_bin', 'train_tag', 'model_idx', 'sort', 'op_name'])[key].idxmin().values
            return self.table().loc[idxmask]
        else:
            idxmask = self.table().groupby(['et_bin', 'eta_bin', 'train_tag', 'model_idx', 'sort', 'op_name'])[key].idxmax().values
            return self.table().loc[idxmask]


    #
    # Get the best sorts from best inits table
    #
    def filter_sorts(self, best_inits, key, idxmin=False):
        '''
        This method will filter the Dataframe based on given key in order to get the best model for every configuration.

        Arguments:

        - key: the column to be used for filter.
        '''
        if idxmin:
            idxmask = best_inits.groupby(['et_bin', 'eta_bin', 'train_tag', 'model_idx', 'op_name'])[key].idxmin().values
            return best_inits.loc[idxmask]
        else:
            idxmask = best_inits.groupby(['et_bin', 'eta_bin', 'train_tag', 'model_idx', 'op_name'])[key].idxmax().values
            return best_inits.loc[idxmask]


    #
    # Calculate the mean/std table from best inits table
    #
    def describe(cls, best_inits ):
        '''
        This method will give the mean and std for construct the beamer presentation for each train tag.

        Arguments:

        - best_inits:
        '''
        # Create a new dataframe to hold this table
        dataframe = { 'train_tag' : [], 'et_bin' : [], 'eta_bin' : [], 'op_name':[],
                      'pd_ref':[], 'fa_ref':[], 'sp_ref':[]}
        # Include all wanted keys into the dataframe
        for key in best_inits.columns.values:
            if key in ['train_tag', 'et_bin', 'eta_bin', 'op_name']:
                continue
            elif 'passed' in key or 'total' in key:
                continue
            elif ('op' in key) or ('val' in key):
                dataframe[key+'_mean'] = []; dataframe[key+'_std'] = []
            else:
                continue

        # Loop over all tuning tags and et/eta bins
        for tag in best_inits.train_tag.unique():
            for et_bin in best_inits.et_bin.unique():
                for eta_bin in best_inits.eta_bin.unique():
                    for op in best_inits.op_name.unique():

                        cv_bin = best_inits.loc[ (best_inits.train_tag == tag) & (best_inits.et_bin == et_bin) & (best_inits.eta_bin == eta_bin) & (best_inits.op_name == op)]
                        dataframe['train_tag'].append( tag )
                        dataframe['et_bin'].append( et_bin )
                        dataframe['eta_bin'].append( eta_bin )
                        dataframe['op_name'].append(op)
                        dataframe['pd_ref'].append(cv_bin.pd_ref.values[0])
                        dataframe['fa_ref'].append(cv_bin.fa_ref.values[0])
                        dataframe['sp_ref'].append(cv_bin.sp_ref.values[0])

                        for key in best_inits.columns.values:
                            if key in ['train_tag', 'et_bin', 'eta_bin', 'op_name']:
                                continue # skip common
                            elif 'passed' in key or 'total' in key:
                                continue # skip counts
                            elif ('op' in key) or ('val' in key):
                                dataframe[key+'_mean'].append( cv_bin[key].mean() ); dataframe[key+'_std'].append( cv_bin[key].std() )
                            else: # skip others
                                continue


        # Return the pandas dataframe
        return pd.DataFrame(dataframe)



    #
    # Get tge cross val integrated table from best inits
    #
    def integrate( self, best_inits, tag ):
        '''
        This method is used to get the integrate information of a given tag.

        Arguments:

        - best_inits: a pandas Dataframe which contains all information for the best inits.
        - tag: the training tag that will be integrate.
        '''
        keys = [ key for key in best_inits.columns.values if 'passed' in key or 'total' in key]
        table = best_inits.loc[best_inits.train_tag==tag].groupby(['sort']).agg(dict(zip( keys, ['sum']*len(keys))))

        for key in keys:
            if 'passed' in key:
                orig_key = key.replace('_passed','')
                values = table[key].div( table[orig_key+'_total'] )
                table[orig_key] = values
                table=table.drop([key],axis=1)
                table=table.drop([orig_key+'_total'],axis=1)
        return table.agg(['mean','std'])


    def evaluate( self, best_sorts, paths , data_generator, dec_generator ):

        tf.config.run_functions_eagerly(False)

        columns = best_sorts.columns.values.tolist()
        extra = ['pd_test', 'fa_test', 'sp_test', 'pd_test_passed', 'pd_test_total', 'fa_test_passed', 'fa_test_total']
        columns.extend(extra)
        table = collections.OrderedDict({ key:[] for key in columns} )

        bins = list(product(range(len(self.__etbins)-1),range(len(self.__etabins)-1)))

        # Loop over all et/eta bins
        for et_bin, eta_bin in tqdm( bins , desc= 'Fitting... ', ncols=70):

                data = data_generator(paths[et_bin][eta_bin])

                for train_tag in best_sorts.train_tag.unique():


                    rows = best_sorts.loc[(best_sorts.et_bin==et_bin) & (best_sorts.eta_bin==eta_bin) & (best_sorts.train_tag==train_tag) ] 
                    
                    for op_name in rows.op_name.unique():

                        row = rows.loc[rows.op_name==op_name]
                        data['dec'] = dec_generator( row, data )

                        pd_test_passed = data.loc[(data.target==1) & (data.dec==True)].shape[0]
                        pd_test_total = data.loc[(data.target==1)].shape[0]
                        pd_test = pd_test_passed/pd_test_total

                        fa_test_passed = data.loc[(data.target!=1) & (data.dec==True)].shape[0]
                        fa_test_total = data.loc[(data.target!=1)].shape[0]
                        fa_test = fa_test_passed/fa_test_total

                        sp_test = np.sqrt(  np.sqrt(pd_test*(1-fa_test)) * (0.5*(pd_test+(1-fa_test)))  )

                        for col in columns:
                            if col in extra:
                                continue
                            table[col].append( getattr(row, col).values[0] )
                        
                        for col in extra:
                            table[col].append(eval(col))

        return pd.DataFrame(table)



    #
    # Dump all history for each line in the table
    #
    def dump_all_history( self, table, output_path , tag):
        '''
        This method will dump the train history. This is a way to get more easy this information when plotting the train evolution.

        Arguments:

        - table: a table with the path information.
        - output_path: the path to sabe the hitories.
        - tag: the train tag.
        '''
        if not os.path.exists( output_path ):
          os.mkdir( output_path )
        for _ , row in table.iterrows():
            if row.train_tag != tag:
              continue
            # Load history
            history = self.get_history(row.file_name, row.tuned_idx)
            history['loc'] = {'et_bin' : row.et_bin, 'eta_bin' : row.eta_bin, 'sort' : row.sort, 'model_idx' : row.model_idx}
            name = 'history_et_%i_eta_%i_model_%i_sort_%i.json' % (row.et_bin,row.eta_bin,row.model_idx,row.sort)
            jbl_name = 'history_et_%i_eta_%i_model_%i_sort_%i.joblib' % (row.et_bin,row.eta_bin,row.model_idx,row.sort)
            joblib.dump(history['summary'], os.path.join(output_path, jbl_name))
            history.pop('summary')
            with open(os.path.join(output_path, '%s' %name), 'w', encoding='utf-8') as fp:
                #json.dump(transform_serialize(history), fp)
                json.dump(str(history), fp)



    def get_history( self, path, index ):
      tuned_list = load(path)['tunedData']
      for tuned in tuned_list:
        if tuned['imodel'] == index:
          return tuned['history']
      MSG_FATAL( self, "It's not possible to find the history for model index %d", index )



	#
	# Plot the training curves for all sorts.
	#
    def plot_training_curves( self, best_inits, best_sorts, dirname, display=False, start_epoch=3 ):
        '''
        This method is a shortcut to plot the monitoring traning curves.

        Arguments:

        - best_inits: a pandas Dataframe which contains all information for the best inits;
        - best_sorts: a pandas Dataframe which contains all information for the best sorts;
        - dirname: a folder to save the figures, if not exist we'll create and attached in $PWD folder;
        - display: a boolean to decide if show or not show the plot;
        - start_epoch: the epoch to start draw the plot.
        '''
        basepath = os.getcwd()
        if not os.path.exists(basepath+'/'+dirname):
          os.mkdir(basepath+'/'+dirname)

        def plot_training_curves_for_each_sort(table, et_bin, eta_bin, best_sort , output, display=False, start_epoch=0):
            table = table.loc[(table.et_bin==et_bin) & (table.eta_bin==eta_bin)]
            nsorts = len(table.sort.unique())
            fig, ax = plt.subplots(nsorts,2, figsize=(15,20))
            fig.suptitle(r'Monitoring Train Plot - Et = %d, Eta = %d'%(et_bin,eta_bin), fontsize=15)
            for idx, sort in enumerate(table.sort.unique()):
                current_table = table.loc[table.sort==sort]
                path=current_table.file_name.values[0]
                history = self.get_history( path, current_table.model_idx.values[0])
                
                best_epoch = history['max_sp_best_epoch_val'][-1] - start_epoch
                # Make the plot here
                ax[idx, 0].set_xlabel('Epochs')
                ax[idx, 0].set_ylabel('Loss (sort = %d)'%sort, color = 'r' if best_sort==sort else 'k')
                ax[idx, 0].plot(history['loss'][start_epoch::], c='b', label='Train Step')
                ax[idx, 0].plot(history['val_loss'][start_epoch::], c='r', label='Validation Step')
                ax[idx, 0].axvline(x=best_epoch, c='k', label='Best epoch')
                ax[idx, 0].legend()
                ax[idx, 0].grid()
                ax[idx, 1].set_xlabel('Epochs')
                ax[idx, 1].set_ylabel('SP (sort = %d)'%sort, color = 'r' if best_sort==sort else 'k')
                ax[idx, 1].plot(history['max_sp_val'][start_epoch::], c='r', label='Validation Step')
                ax[idx, 1].axvline(x=best_epoch, c='k', label='Best epoch')
                ax[idx, 1].legend()
                ax[idx, 1].grid()

            plt.savefig(output)
            if display:
                plt.show()
            else:
                plt.close(fig)

        for tag in best_inits.train_tag.unique():
            for et_bin in best_inits.et_bin.unique():
                for eta_bin in best_inits.eta_bin.unique():
                    best_sort = best_sorts.loc[(best_sorts.et_bin==et_bin) & (best_sorts.eta_bin==eta_bin) & (best_sorts.train_tag==tag)].sort 
                    plot_training_curves_for_each_sort(best_inits.loc[best_inits.train_tag==tag], et_bin, eta_bin, best_sort.values[0],
                        basepath+'/'+dirname+'/train_evolution_et%d_eta%d_%s.pdf'%(et_bin,eta_bin,tag), display, start_epoch)



    #
    # Plot the training curves for all sorts.
    #
    def plot_roc_curves( self, best_sorts, tags, legends, output, display=False, colors=None, points=None, et_bin=None, eta_bin=None,
                         xmin=-0.02, xmax=0.3, ymin=0.8, ymax=1.02, fontsize=18, figsize=(15,15)):
        '''
        This method will plot the ROC curves.

        Arguments:

        - best_sorts: a pandas Dataframe which contains all information for the best sorts;
        - tag: the tuning tags to be plotted
        - legends: the legends to used (unused)
        - output: the path with the name to save the plot;
        - display: a boolean to decide if show or not show the plot;
        - colors: the color list to use in plot;
        - points: the points list to diferenciate the curves (unused);
        - et_bin: the et bin index
        - eta_bin: the eta bin index
        - xmin: x plot lower limit;
        - xmax: x plot upper limit;
        - ymin: y plot lower limit;
        - ymax: y plot upper limit;
        - fontsize: font size just like in matplotlib
        - figsize: figure size just like in matplotlib
        '''

        def plot_roc_curves_for_each_bin(ax, table, colors, xmin=-0.02, xmax=0.3, ymin=0.8, ymax=1.02, fontsize=18):

          ax.set_xlabel('Fake Probability [%]',fontsize=fontsize)
          ax.set_ylabel('Detection Probability [%]',fontsize=fontsize)
          ax.set_title(r'Roc curve (et = %d, eta = %d)'%(table.et_bin.values[0], table.eta_bin.values[0]),fontsize=fontsize)

          for idx, tag in enumerate(tags):
              current_table = table.loc[(table.train_tag==tag)]

              path=current_table.file_name.values[0]
              history = self.get_history( path, current_table.model_idx.values[0])
              pd, fa = history['summary']['rocs']['roc_op']
              ax.plot( fa, pd, color=colors[idx], linewidth=2, label=tag)
              ax.set_ylim(ymin,ymax)
              ax.set_xlim(xmin,xmax)

          ax.legend(fontsize=fontsize)
          ax.grid()


        if et_bin!=None and eta_bin!=None:
            fig, ax = plt.subplots(1,1, figsize=figsize)
            fig.suptitle(r'Operation ROCs', fontsize=15)
            table_for_this_bin = best_sorts.loc[(best_sorts.et_bin==et_bin) & (best_sorts.eta_bin==eta_bin)]
            plot_roc_curves_for_each_bin( ax, table_for_this_bin, colors, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, fontsize=fontsize)
            plt.savefig(output)
            if display:
                plt.show()
            else:
                plt.close(fig)

        else:

            n_et_bins = len(best_sorts.et_bin.unique())
            n_eta_bins = len(best_sorts.eta_bin.unique())
            fig, ax = plt.subplots(n_et_bins,n_eta_bins, figsize=figsize)
            fig.suptitle(r'Operation ROCs', fontsize=15)

            for et_bin in best_sorts.et_bin.unique():
                for eta_bin in best_sorts.eta_bin.unique():
                    table_for_this_bin = best_sorts.loc[(best_sorts.et_bin==et_bin) & (best_sorts.eta_bin==eta_bin)]
                    plot_roc_curves_for_each_bin( ax[et_bin][eta_bin], table_for_this_bin, colors, xmin=xmin, xmax=xmax,
                                                  ymin=ymin, ymax=ymax, fontsize=fontsize)

            plt.savefig(output)
            if display:
                plt.show()
            else:
                plt.close(fig)




    #
    # Create the beamer table file
    #
    def dump_beamer_table( self, best_inits, output, tags=None, title='' , test_table=None):
        '''
        This method will use a pandas Dataframe in order to create a beamer presentation which summary the tuning cross-validation.

        Arguments:
        - best_inits: a pandas Dataframe which contains all information for the best inits.
        - operation_poins: the operation point that will be used.
        - output: a name for the pdf
        - tags: the training tag that will be used. If None then the tags will be get from the Dataframe.
        - title: the pdf title
        '''

        cv_table = self.describe( best_inits )
        # Create Latex Et bins
        etbins_str = []; etabins_str = []
        for etBinIdx in range(len(self.__etbins)-1):
            etbin = (self.__etbins[etBinIdx], self.__etbins[etBinIdx+1])
            if etbin[1] > 100 :
                etbins_str.append( r'$E_{T}\text{[GeV]} > %d$' % etbin[0])
            else:
                etbins_str.append(  r'$%d < E_{T} \text{[Gev]}<%d$'%etbin )

        # Create Latex eta bins
        for etaBinIdx in range( len(self.__etabins)-1 ):
            etabin = (self.__etabins[etaBinIdx], self.__etabins[etaBinIdx+1])
            etabins_str.append( r'$%.2f<\eta<%.2f$'%etabin )

        # Default colors
        colorPD = '\\cellcolor[HTML]{9AFF99}'; colorPF = ''; colorSP = ''

        train_tags = cv_table.train_tag.unique() if not tags else tags

        # fix tags to list
        if type(tags) is str: tags=[tags]

        # Apply beamer
        with BeamerTexReportTemplate1( theme = 'Berlin'
                                 , _toPDF = True
                                 , title = title
                                 , outputFile = output
                                 , font = 'structurebold' ):

                ### Prepare tables
                tuning_names = ['']; tuning_names.extend( train_tags )
                lines1 = []
                lines1 += [ HLine(_contextManaged = False) ]
                lines1 += [ HLine(_contextManaged = False) ]
                lines1 += [ TableLine( columns = ['','','kinematic region'] + reduce(lambda x,y: x+y,[['',s,''] for s in etbins_str]), _contextManaged = False ) ]
                lines1 += [ HLine(_contextManaged = False) ]
                lines1 += [ TableLine( columns = ['Det. Region','Method','Type'] + reduce(lambda x,y: x+y,[[colorPD+r'$P_{D}[\%]$',colorSP+r'$SP[\%]$',colorPF+r'$P_{F}[\%]$'] \
                                      for _ in etbins_str]), _contextManaged = False ) ]
                lines1 += [ HLine(_contextManaged = False) ]

                for etaBinIdx in cv_table.eta_bin.unique():
                    for idx, tag in enumerate( train_tags ):
                        cv_values=[]; ref_values=[]; test_values=[]
                        for etBinIdx in cv_table.et_bin.unique():
                            current_table = cv_table.loc[ (cv_table.train_tag==tag) & (cv_table.et_bin==etBinIdx) & (cv_table.eta_bin==etaBinIdx) ]
                            sp = current_table['sp_val_mean'].values[0]*100
                            pd = current_table['pd_val_mean'].values[0]*100
                            fa = current_table['fa_val_mean'].values[0]*100
                            sp_std = current_table['sp_val_std'].values[0]*100
                            pd_std = current_table['pd_val_std'].values[0]*100
                            fa_std = current_table['fa_val_std'].values[0]*100
                            sp_ref = current_table['sp_ref'].values[0]*100
                            pd_ref = current_table['pd_ref'].values[0]*100
                            fa_ref = current_table['fa_ref'].values[0]*100

                            cv_values   += [ colorPD+('%1.2f$\pm$%1.2f')%(pd,pd_std),colorSP+('%1.2f$\pm$%1.2f')%(sp,sp_std),colorPF+('%1.2f$\pm$%1.2f')%(fa,fa_std),    ]
                            ref_values  += [ colorPD+('%1.2f')%(pd_ref), colorSP+('%1.2f')%(sp_ref), colorPF+('%1.2f')%(fa_ref)]
                            
                            if test_table is not None:
                                current_test_table = test_table.loc[ (test_table.train_tag==tag) & (test_table.et_bin==etBinIdx) & (test_table.eta_bin==etaBinIdx) ]
                                pd = current_test_table['pd_test'].values[0]*100
                                sp = current_test_table['sp_test'].values[0]*100
                                fa = current_test_table['fa_test'].values[0]*100
                                test_values   += [ colorPD+('%1.2f')%(pd),colorSP+('%1.2f')%(sp),colorPF+('%1.2f')%(fa)    ]

                        if idx > 0:
                            lines1 += [ TableLine( columns = ['', tuning_names[idx+1], 'Cross Val.'] + cv_values   , _contextManaged = False ) ]
                        
                        
                            if test_table is not None:
                                lines1 += [ TableLine( columns = ['', tuning_names[idx+1], 'Test'] + test_values   , _contextManaged = False ) ]
                    
                        
                        else:
                            lines1 += [ TableLine( columns = ['\multirow{%d}{*}{'%(len(tuning_names))+etabins_str[etaBinIdx]+'}',tuning_names[idx], 'Reference'] + ref_values   ,
                                                _contextManaged = False ) ]
                            lines1 += [ TableLine( columns = ['', tuning_names[idx+1], 'Cross Val.'] + cv_values    , _contextManaged = False ) ]
                            if test_table is not None:
                                lines1 += [ TableLine( columns = ['', tuning_names[idx+1], 'Test'] + test_values   , _contextManaged = False ) ]
                    
                    lines1 += [ HLine(_contextManaged = False) ]
                lines1 += [ HLine(_contextManaged = False) ]


                ### Calculate the final efficiencies
                lines2 = []
                lines2 += [ HLine(_contextManaged = False) ]
                lines2 += [ HLine(_contextManaged = False) ]
                lines2 += [ TableLine( columns = ['',colorPD+r'$P_{D}[\%]$',colorPF+r'$F_{a}[\%]$'], _contextManaged = False ) ]
                lines2 += [ HLine(_contextManaged = False) ]
                for idx, tag in enumerate( train_tags ):
                    current_table = self.integrate( best_inits, tag )
                    pd = current_table['pd_val'].values[0]*100
                    pd_std = current_table['pd_val'].values[1]*100
                    fa = current_table['fa_val'].values[0]*100
                    fa_std = current_table['fa_val'].values[1]*100

                    pd_ref = current_table['pd_ref'].values[0]*100
                    fa_ref = current_table['fa_ref'].values[0]*100

                    if test_table is not None:
                        keys = [ key for key in test_table.columns.values if 'passed' in key or 'total' in key]
                        current_test_table = test_table.loc[test_table.train_tag=='v8'].agg(dict(zip( keys, ['sum']*len(keys))))
                        test_pd = (current_test_table['pd_test_passed']/current_test_table['pd_test_total'])*100
                        test_fa = (current_test_table['fa_test_passed']/current_test_table['fa_test_total'])*100

                    if idx > 0:
                        lines2 += [ TableLine( columns = [tag.replace('_','\_') + ' (Cross Val.)' ,
                          colorPD+('%1.2f$\pm$%1.2f')%(pd,pd_std),colorPF+('%1.2f$\pm$%1.2f')%(fa,fa_std) ], _contextManaged = False ) ]
                    
                        if test_table is not None:
                            lines2 += [ TableLine( columns = [tag.replace('_','\_') + ' (Test)',
                            colorPD+('%1.2f')%(test_pd),colorPF+('%1.2f')%(test_fa) ], _contextManaged = False ) ]
                    
                    else:
                        lines2 += [ TableLine( columns = ['Ref.' ,colorPD+('%1.2f')%(pd_ref),colorPF+('%1.2f')%(fa_ref) ], _contextManaged = False ) ]
                        lines2 += [ TableLine( columns = [tag.replace('_','\_') + ' (Cross Val.)',
                                    colorPD+('%1.2f$\pm$%1.2f')%(pd,pd_std),colorPF+('%1.2f$\pm$%1.2f')%(fa,fa_std) ], _contextManaged = False ) ]
                        if test_table is not None:
                            lines2 += [ TableLine( columns = [tag.replace('_','\_')+ ' (Test)' ,
                            colorPD+('%1.2f')%(test_pd),colorPF+('%1.2f')%(test_fa) ], _contextManaged = False ) ]
                    

                # Create all tables into the PDF Latex
                with BeamerSlide( title = "The Cross Validation Efficiency Values For All Tunings"  ):
                    with Table( caption = 'The $P_{d}$, $F_{a}$ and $SP $ values for each phase space for each method.') as table:
                        with ResizeBox( size = 1. ) as rb:
                            with Tabular( columns = '|lcc|' + 'ccc|' * len(etbins_str) ) as tabular:
                                tabular = tabular
                                for line in lines1:
                                    if isinstance(line, TableLine):
                                        tabular += line
                                    else:
                                        TableLine(line, rounding = None)

                with BeamerSlide( title = "The General Efficiency"  ):
                    with Table( caption = 'The general efficiency for the cross validation method for each method.') as table:
                        with ResizeBox( size = 0.7 ) as rb:
                            with Tabular( columns = 'lc' + 'c' * 2 ) as tabular:
                                tabular = tabular
                                for line in lines2:
                                    if isinstance(line, TableLine):
                                        tabular += line
                                    else:
                                        TableLine(line, rounding = None)



    #
    # Load all keras models given the best sort table
    #
    def get_best_models( self, best_sorts , remove_last=True, with_history=True):
        '''
        This method will load the best models.

        Arguments:

        - best_sorts: the table that contains the best_sorts;
        - remove_last: a bolean variable to remove or not the tanh in tha output layer;
        - with_history: unused variable.
        '''
        from tensorflow.keras.models import Model, model_from_json
        import json

        models = [[ None for _ in range(len(self.__etabins)-1)] for __ in range(len(self.__etbins)-1)]
        for et_bin in range(len(self.__etbins)-1):
            for eta_bin in range(len(self.__etabins)-1):
                d_tuned = {}
                best = best_sorts.loc[(best_sorts.et_bin==et_bin) & (best_sorts.eta_bin==eta_bin)]
                tuned = load(best.file_name.values[0])['tunedData'][best.tuned_idx.values[0]]
                model = model_from_json( json.dumps(tuned['sequence'], separators=(',', ':')) ) #custom_objects={'RpLayer':RpLayer} )
                model.set_weights( tuned['weights'] )
                new_model = Model(model.inputs, model.layers[-2].output) if remove_last else model
                #new_model.summary()
                d_tuned['model']    = new_model
                d_tuned['etBin']    = [self.__etbins[et_bin], self.__etbins[et_bin+1]]
                d_tuned['etaBin']   = [self.__etabins[eta_bin], self.__etabins[eta_bin+1]]
                d_tuned['etBinIdx'] = et_bin
                d_tuned['etaBinIdx']= eta_bin
                d_tuned['sort']     = best.sort.values[0]
                d_tuned['init']     = best.init.values[0]
                d_tuned['model_idx']= best.model_idx.values[0]
                d_tuned['file_name']= best.file_name.values[0]
                if with_history:
                    d_tuned['history']  = tuned['history']
                models[et_bin][eta_bin] = d_tuned
        return models





    #
    # Load all keras models given the best sort table
    #
    def get_best_init_models( self, best_inits , remove_last=True, with_history=True):
        '''
        This method will load the best init models.

        Arguments:

        - best_inits: the table that contains the best_inits;
        - remove_last: a bolean variable to remove or not the tanh in tha output layer;
        - with_history: unused variable.
        '''
        from tensorflow.keras.models import Model, model_from_json
        import json

        models = [[ [] for _ in range(len(self.__etabins)-1)] for __ in range(len(self.__etbins)-1)]

        for et_bin in range(len(self.__etabins)-1):
            for eta_bin in range(len(self.__etabins)-1):
                for sort in best_inits.sort.unique():
                    d_tuned = {}
                    best = best_inits.loc[(best_inits.et_bin==et_bin) & (best_inits.eta_bin==eta_bin) & (best_inits.sort==sort)]
                    tuned = load(best.file_name.values[0])['tunedData'][best.tuned_idx.values[0]]
                    model = model_from_json( json.dumps(tuned['sequence'], separators=(',', ':')) ) #custom_objects={'RpLayer':RpLayer} )
                    model.set_weights( tuned['weights'] )
                    new_model = Model(model.inputs, model.layers[-2].output) if remove_last else model
                    #new_model.summary()
                    d_tuned['model']    = new_model
                    d_tuned['etBin']    = [self.__etbins[et_bin], self.__etbins[et_bin+1]]
                    d_tuned['etaBin']   = [self.__etabins[eta_bin], self.__etabins[eta_bin+1]]
                    d_tuned['etBinIdx'] = et_bin
                    d_tuned['etaBinIdx']= eta_bin
                    d_tuned['sort']     = best.sort.values[0]
                    d_tuned['init']     = best.init.values[0]
                    d_tuned['model_idx']= best.model_idx.values[0]
                    d_tuned['file_name']= best.file_name.values[0]
                    if with_history:
                        d_tuned['history']  = tuned['history']
                    models[et_bin][eta_bin].append(d_tuned)
        return models
