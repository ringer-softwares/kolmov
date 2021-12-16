
__all__ = [
    'get_color_fader',
    'training_curves', # should be training_curves for future?
    'plot_quadrant' , # should be plot_quadrant for future?
    'plot_correction', 
]

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import rootplotlib as rpl
import array

from Gaugi import Logger, expand_folders
from Gaugi.macros import *
from copy import copy

from kolmov.utils.constants import str_etbins_zee, str_etabins
from ROOT import kBlackBody, TCanvas, TGraphErrors, TLine, gStyle, kBlue, kBlack



def get_color_fader( c1, c2, n ):
    def color_fader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        c1=np.array(mpl.colors.to_rgb(c1))
        c2=np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
    return [ color_fader(c1,c2, frac) for frac in np.linspace(0,1,n) ]




class training_curves( Logger ):


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
        - str_et_bins: a list which contains the et boundaries.
        The default values are the zee et binning, other are in constants.
        '''

        Logger.__init__( self )
        self.plot_names = {
                            'loss' : 'Loss Function Evolution',
                            'sp'   : r'$SP$ Index Evolution',
                            'pd'  : r'$P_D$ Evolution',
                            'fa'  : r'$F_A$ Evolution',
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
            if path.split('.')[-1] != 'json':
                continue
            
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
        if not os.path.exists( plot_path ):
          os.makedirs( plot_path )

        # a easy way to lock only in the necessaries keys
        wanted_h_keys = ['et%i_eta%i_sort_%i' %(et_bin, eta_bin, i) for i in range(10)]


        # train indicators (today we use only this)
        train_indicators = [('loss', 'val_loss'), ('sp', 'val_sp'), ('pd', 'val_pd'), ('fa', 'val_fa')]
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
                    ax[idx, jdx].axvline(x=len(self.h_dict[isort][train_ind[1]])-25, label='Best Epoch',
                                         c='black', lw=1.25, ls='--')
                    ax[idx, jdx].legend()
                    ax[idx, jdx].grid()
                else:
                    ax[idx, jdx].set_title('%s - %s' %(self.plot_names[train_ind],
                                                       self.sort_name_dict[isort_k]))
                    ax[idx, jdx].axvline(x=self.h_dict[isort]['max_sp_best_epoch_val'][-1], label='Best Epoch',
                                         c='black', lw=1.25, ls='--')
                    ax[idx, jdx].set_xlabel('Epochs')
                    ax[idx, jdx].plot(self.h_dict[isort][train_ind], c='g', label='Validation Step')
                    ax[idx, jdx].grid()
        # set the figure name
        fig.suptitle(r'Monitoring Train Plot - $E_T$: %s | $\eta$: %s'
                     %(self.et_range[et_bin], self.eta_range[eta_bin]), fontsize=20)
        # save figure
        fig.savefig(os.path.join(plot_path, '%s.png' %(plot_name)),
                    dpi=300,
                    bbox_inches='tight', facecolor='white')
        plt.close();
        return




#
# plot quadrant histograms
#
def plot_quadrant(df, plot_configs, output_path='/volume'):
    '''
    This function will provide a simple implementation of prometheus quadrant analysis
    applying conditions on Dataframe.

    Arguments:
    - df: the Dataframe used to extract information;
    - plot_configs: a dictionary which contains the all plot information and the
    condition to produce the plot;
    - outpu_path: the path to save the plot.
    '''
    for ivar in plot_configs.keys():
        print('Plotting %s... ' %(ivar))
        local_config = plot_configs[ivar]
        # loop over variables
        var_cond1 = df[((local_config['cond1']) &\
                        (local_config['common_cond']))][ivar].values
        var_cond1 = var_cond1[((var_cond1 >= local_config['low_edge']) &\
                               (var_cond1 <= local_config['high_edge']))]

        var_cond2 = df[((local_config['cond2']) &\
                        (local_config['common_cond']))][ivar].values
        var_cond2 = var_cond2[((var_cond2 >= local_config['low_edge']) &\
                               (var_cond2 <= local_config['high_edge']))]

        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6))

        fig.suptitle(r'%s Distribution Comparision - %s | %s' %(local_config['var_name'],
                                                                local_config['title_tag1'],
                                                                local_config['title_tag2']),
                    fontsize=20)

        ns, bins, patches =  ax1.hist([var_cond1, var_cond2],
                                    bins=local_config['nbins'],
                                    color=['blue', 'red'],
                                    histtype='step',
                                    alpha=.5,
                                    label=[local_config['cond1_label'],
                                           local_config['cond2_label']],
                                    lw=2)
        ax1.legend(loc='upper left')

        # ratio plot - vulgo puchadinho
        ax2.plot(bins[:-1],     # this is what makes it comparable
                100*(ns[1] / ns[0]), # maybe check for div-by-zero!
                '^',
                color='black')


        ax1.set_ylabel('Counts', fontsize=15)
        ax2.set_ylabel('Ratio (%)', fontsize=15)
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel(r'%s' %(local_config['var_name']), fontsize=15)
        # at this time we don't need this
        # ax1.set_xticks(np.arange(start=-0.05, stop=0.06, step=1e-2))
        # ax2.set_xticks(np.arange(start=-0.05, stop=0.06, step=1e-2))
        ax1.set_xlim([local_config['low_edge'], local_config['high_edge']])
        ax2.set_xlim([local_config['low_edge'], local_config['high_edge']])
        ax2.grid()
        ax1.grid()
        plt.savefig(os.path.join(output_path,
                                '%s_mini_quad.png' %(ivar)),
                    dpi=300)





#
# Plot 2D histogram function based on ROOT used by fit_table class
#
def plot_correction( h2, slope, offset, x_points, y_points, error_points, outname, 
                     xlabel='',
                     etBinIdx=None, 
                     etaBinIdx=None, 
                     etBins=None,
                     etaBins=None, 
                     label='Internal',
                     ref_value=None, 
                     pd_value=None,
                     palette=kBlackBody):

    def toStrBin(etlist = None, etalist = None, etidx = None, etaidx = None):
        if etlist and etidx is not None:
            etlist=copy(etlist)
            if etlist[-1]>9999:  etlist[-1]='#infty'
            binEt = (str(etlist[etidx]) + ' < E_{T} [GeV] < ' + str(etlist[etidx+1]) if etidx+1 < len(etlist) else
                                     'E_{T} > ' + str(etlist[etidx]) + ' GeV')
            return binEt
        if etalist and etaidx is not None:
            binEta = (str(etalist[etaidx]) + ' < |#eta| < ' + str(etalist[etaidx+1]) if etaidx+1 < len(etalist) else
                                        str(etalist[etaidx]) + ' <|#eta| < 2.47')
            return binEta

    canvas = TCanvas("canvas","canvas",500,500)
    rpl.set_figure(canvas)
    gStyle.SetPalette(palette)
    canvas.SetRightMargin(0.15)
    canvas.SetTopMargin(0.15)
    canvas.SetLogz()
    h2.GetXaxis().SetTitle('Neural Network output (Discriminant)')
    h2.GetYaxis().SetTitle(xlabel)
    h2.GetZaxis().SetTitle('Count')
    h2.Draw('colz')
    g = TGraphErrors(len(x_points), array.array('d',x_points), array.array('d',y_points), array.array('d',error_points), array.array('d',[0]*len(x_points)))
    g.SetMarkerColor(kBlue)
    g.SetMarkerStyle(8)
    g.SetMarkerSize(1)
    g.Draw("P same")
    line = TLine(slope*h2.GetYaxis().GetXmin()+offset,h2.GetYaxis().GetXmin(), slope*h2.GetYaxis().GetXmax()+offset, h2.GetYaxis().GetXmax())
    line.SetLineColor(kBlack)
    line.SetLineWidth(2)
    line.Draw()
    # Add text labels into the canvas
    text = toStrBin(etlist=etBins, etidx=etBinIdx)
    text+= ', '+toStrBin(etalist=etaBins, etaidx=etaBinIdx)
    if ref_value and pd_value:
        text+=', P_{D} = %1.2f (%1.2f) [%%]'%(pd_value*100, ref_value*100)
    rpl.add_text(0.15, 0.885, text, textsize=.03)
    rpl.set_atlas_label(0.15, 0.94, label)
    rpl.format_canvas_axes(XLabelSize=16, YLabelSize=16, XTitleOffset=0.87, ZLabelSize=16,ZTitleSize=16, YTitleOffset=0.87, ZTitleOffset=1.1)
    canvas.SaveAs(outname)
