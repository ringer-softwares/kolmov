
#
# DISCLAIMER: You should have root installed at your system to use it.
#

__all__ = ['crossval_fit_table']

from Gaugi import Logger, LoggingLevel, StoreGate, restoreStoreGate
from Gaugi.macros import *
from Gaugi import load as gload

from tqdm import tqdm
#from tqdm import tqdm_notebook as tqdm


from copy import copy
from itertools import product
from pprint import pprint

import collections
import numpy as np
import pandas as pd
import time
import array
import json
import collections
import os
import rootplotlib as rpl

from kolmov.utils.plot_functions import plot_correction


import ROOT
from ROOT import kBird,kBlackBody,kTRUE
from ROOT import TCanvas, gStyle, TLegend, gPad, TLatex, TEnv, gROOT, TLine, TColor
from ROOT import kAzure, kRed, kBlue, kBlack,kBird, kOrange,kGray
from ROOT import TGraphErrors,TF1,TH1F,TH2F


from pybeamer import *
from tensorflow.keras.models import Model, model_from_json


rpl.set_atlas_style()
rpl.suppress_root_warnings()
gROOT.SetBatch(kTRUE)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def my_model_generator( row, remove_last=True):
    file_name = row.file_name.values[0]
    tuned_idx = row.tuned_idx.values[0]
    model_idx = row.model_idx.values[0]
    tuned = gload(file_name)['tunedData'][tuned_idx]
    model = model_from_json( json.dumps(tuned['sequence'], separators=(',', ':')) ) #custom_objects={'RpLayer':RpLayer} )
    model.set_weights( tuned['weights'] )
    if remove_last:
        model = Model(model.inputs, model.layers[-2].output)
    return model, file_name, tuned_idx, model_idx

   


#
# Linear correction class table
#
class crossval_fit_table(Logger):

    #
    # Constructor
    #
    def __init__(self, etbins, etabins, kf, data_generator, input_generator,
                 model_generator=my_model_generator):

        # init base class
        Logger.__init__(self)
        self.etbins = etbins
        self.etabins = etabins
        self.__data_generator = data_generator
        self.__input_generator = input_generator
        self.__kf = kf
        self.__model_generator = model_generator




    def fill( self, best_inits, paths, ref_values, xbin_size=0.05, ybin_size=0.5, 
              ymin=16, ymax=60, xmin_percentage = 1, xmax_percentage=99, output_path='', 
              min_avgmu=16, max_avgmu=100, false_alarm_limit=0.5):
        
        tf.config.run_functions_eagerly(False)

        output_path += '/figures'
        try:
            if not os.path.exists(output_path): os.makedirs(output_path)
        except:
            MSG_WARNING( self,'The director %s exist.', output_path)

        columns = best_inits.columns.values.tolist()
        columns.extend([ 
                          'slope',
                          'offset', 
                          'fig_signal', 
                          'fig_background',
                        ])

        dataframe = collections.OrderedDict({col:[] for col in columns})


        for et_bin in best_inits.et_bin.unique():

            for eta_bin in best_inits.eta_bin.unique():

                best_inits_bin = best_inits.loc[(best_inits.et_bin==et_bin)&(best_inits.eta_bin==eta_bin)]

                # check if the current table has this phase space in
                if ( best_inits_bin.shape[0] == 0) :
                    continue

                ref = ref_values[et_bin][eta_bin]

                # read the phase space data to the memory
                data_df = self.__data_generator( paths[et_bin][eta_bin] )
                splits = [(train_idx, val_idx) for train_idx,val_idx in self.__kf.split(data_df, data_df.target)]

                # Loop over all sorts
                for sort in tqdm(best_inits_bin.sort.unique(), 
                                 desc='Running (et=%d, eta%d)'%(et_bin,eta_bin) ):

                    train_idx = splits[sort][0]
                    val_idx   = splits[sort][1]

                    # get model
                    model, file_name, tuned_idx, model_idx = self.__model_generator(best_inits_bin.loc[best_inits_bin.sort==sort])
                    output = model.predict( self.__input_generator(data_df), batch_size=4096).flatten()
                    avgmu  = data_df.avgmu.values.flatten()
                    target = data_df.target.values.flatten()

                    # prepare all histograms
                    xmin = int(np.percentile(output , xmin_percentage))
                    xmax = int(np.percentile(output , xmax_percentage))

                    xbins = int((xmax-xmin)/xbin_size)
                    ybins = int((ymax-ymin)/ybin_size)

                    df_train = pd.DataFrame({ 'output' : output[train_idx], 
                                              'avgmu'  : avgmu[train_idx],
                                              'target' : target[train_idx]})     

                    # Fill train set
                    hist_signal = rpl.hist2d.make_hist('signal', 
                                                        df_train.loc[df_train.target==1].output.values, 
                                                        df_train.loc[df_train.target==1].avgmu.values, 
                                                        xbins, xmin, xmax, ybins, ymin, ymax)

                    hist_background = rpl.hist2d.make_hist('background',
                                                           df_train.loc[df_train.target!=1].output.values, 
                                                           df_train.loc[df_train.target!=1].avgmu.values, 
                                                           xbins, xmin, xmax, ybins, ymin, ymax)

                    if min_avgmu:
                        avgmu[avgmu < min_avgmu] = min_avgmu
                    if max_avgmu:
                        avgmu[avgmu > max_avgmu] = max_avgmu


                    df_train['avgmu'] = avgmu[train_idx]

                    df_val = pd.DataFrame({ 'output':output[val_idx], 
                                            'avgmu' :avgmu[val_idx],
                                            'target':target[val_idx]})

                    df_op  = pd.DataFrame({ 'output':output, 
                                            'avgmu' :avgmu,
                                            'target':target})

                    # Loop over all operation points
                    for op_name in best_inits_bin.op_name.unique():
                    
                        row = best_inits_bin.loc[ (best_inits_bin.sort==sort) &
                                                  (best_inits_bin.op_name==op_name) ]
                        init = row.init.values[0]
                        file_name = row.file_name.values[0] 

                        pd_ref = ref[op_name]['pd']; fa_ref = ref[op_name]['fa']
                        sp_ref  = np.sqrt(  np.sqrt(pd_ref*(1-fa_ref)) * (0.5*(pd_ref+(1-fa_ref)))  )

                        train_tag      = row.train_tag.values[0]
                        max_sp_val     = row.max_sp_val.values[0]
                        max_sp_pd_val  = row.max_sp_pd_val.values[0]
                        max_sp_fa_val  = row.max_sp_fa_val.values[0]
                        max_sp_op      = row.max_sp_op.values[0]
                        max_sp_pd_op   = row.max_sp_pd_op.values[0]
                        max_sp_fa_op   = row.max_sp_fa_op.values[0]
                        pd_ref_passed  = row.pd_ref_passed.values[0]
                        fa_ref_passed  = row.fa_ref_passed.values[0]
                        pd_ref_total   = row.pd_ref_total.values[0]
                        fa_ref_total   = row.fa_ref_total.values[0]

                        #
                        # Train events (calculate slope and offset)
                        #

                        # calculate slope and offset from the train set
                        slope, offset, info = self.calculate( hist_signal, hist_background, pd_ref, false_alarm_limit=false_alarm_limit )
                        df_train['dec'] = np.greater(df_train.output, slope*df_train.avgmu + offset)
                        pd_train_passed = df_train.loc[(df_train.target==True) & (df_train.dec==True)].shape[0]
                        pd_train_total  = df_train.loc[(df_train.target==True)].shape[0]
                        pd_train        = pd_train_passed/pd_train_total

                        #
                        # Validation
                        #

                        df_val['dec'] = np.greater(df_val.output.values, slope*df_val.avgmu.values + offset)
                        # signal (val)
                        pd_val_passed = df_val.loc[(df_val.target==True) & (df_val.dec==True)].shape[0]
                        pd_val_total  = df_val.loc[(df_val.target==True)].shape[0]
                        pd_val = pd_val_passed/pd_val_total

                        # background (val)
                        fa_val_passed = df_val.loc[(df_val.target==False) & (df_val.dec==True)].shape[0]
                        fa_val_total  = df_val.loc[(df_val.target==False)].shape[0]
                        fa_val = fa_val_passed/fa_val_total
                        sp_val = np.sqrt(  np.sqrt(pd_val*(1-fa_val)) * (0.5*(pd_val+(1-fa_val)))  )

                        #
                        # Operation (Train + val)
                        #

                        df_op['dec'] = np.greater(df_op.output, slope*df_op.avgmu + offset)
                        # signal (train)
                        pd_op_passed = df_op.loc[(df_op.target==True) & (df_op.dec==True)].shape[0]
                        pd_op_total  = df_op.loc[(df_op.target==True)].shape[0]
                        pd_op  = pd_op_passed/pd_op_total
                        # background (train)
                        fa_op_passed = df_op.loc[(df_op.target==False) & (df_op.dec==True)].shape[0]
                        fa_op_total  = df_op.loc[(df_op.target==False)].shape[0]
                        fa_op  = fa_op_passed/fa_op_total
                        sp_op  = np.sqrt(  np.sqrt(pd_op*(1-fa_op)) * (0.5*(pd_op+(1-fa_op)))  )

                        #
                        # Plot histograms
                        #

                        label='Internal, #it{%s}'%op_name

                        fig_signal = output_path+'/correction_signal_%s_et%d_eta%d_sort%d.pdf'%(op_name,et_bin,eta_bin,sort)
                        plot_correction( hist_signal, slope, offset, info['x_dots'], info['y_dots'], info['errors'], 
                                         fig_signal, xlabel='<#mu>',
                                         etBinIdx=et_bin, etaBinIdx=eta_bin, 
                                         etBins=self.etbins,etaBins=self.etabins,
                                         label=label, ref_value=pd_ref, 
                                         pd_value=pd_train)

                        fig_background = output_path+'/correction_background_%s_et%d_eta%d_sort%d.pdf'%(op_name,et_bin,eta_bin,sort)
                        plot_correction( hist_background, slope, offset, info['x_dots'], info['y_dots'], info['errors'], 
                                         fig_background, xlabel='<#mu>',
                                         etBinIdx=et_bin, etaBinIdx=eta_bin, 
                                         etBins=self.etbins,etaBins=self.etabins,
                                         label=label, ref_value=pd_ref, 
                                         pd_value=pd_train)


                        # hold all values into the table
                        for col in columns:
                            dataframe[col].append(eval(col))


        self.__table =  pd.DataFrame(dataframe)
        return self.__table

    #
    # Get table
    #
    def table(self):
        return self.__table

    #
    # Fill correction table
    #
    def calculate( self, hist_signal, hist_background, pd_ref, false_alarm_limit = 0.5):

        d = {'x_dots' : [], 'y_dots':[], 'errors':[]}

        false_alarm = 1.0
        while false_alarm > false_alarm_limit:

            # Get the threshold when we not apply any linear correction
            threshold, _ = self.find_threshold( hist_signal.ProjectionX(), pd_ref )
            # Get the efficiency without linear adjustment
            signal_eff, signal_num, signal_den = self.calculate_num_and_den_from_hist(hist_signal, 0.0, threshold)
            
            # Apply the linear adjustment and fix it in case of positive slope
            slope, offset, x_dots, y_dots, errors = self.fit( hist_signal, pd_ref )

            if slope > 0:
                slope = 0; offset = threshold

            # Get the efficiency with linear adjustment
            signal_corrected_eff, signal_corrected_num, signal_corrected_den = self.calculate_num_and_den_from_hist(hist_signal, slope, offset)
            false_alarm, background_corrected_num, background_corrected_den = self.calculate_num_and_den_from_hist(hist_background, slope, offset)

            if false_alarm > false_alarm_limit:
                # Reduce the reference value by hand
                pd_ref-=0.0025


            d['x_dots'] = x_dots
            d['y_dots'] = y_dots
            d['errors'] = errors
        
        return slope, offset, d


    #
    # open the tuned model from npz file
    #

    #
    # Find the threshold given a reference value
    #
    def find_threshold(self, th1,effref):
        nbins = th1.GetNbinsX()
        fullArea = th1.Integral(0,nbins+1)
        if fullArea == 0:
            return 0,1
        notDetected = 0.0; i = 0
        while (1. - notDetected > effref):
            cutArea = th1.Integral(0,i)
            i+=1
            prevNotDetected = notDetected
            notDetected = cutArea/fullArea
        eff = 1. - notDetected
        prevEff = 1. -prevNotDetected
        deltaEff = (eff - prevEff)
        threshold = th1.GetBinCenter(i-1)+(effref-prevEff)/deltaEff*(th1.GetBinCenter(i)-th1.GetBinCenter(i-1))
        error = 1./np.sqrt(fullArea)
        return threshold, error

    #
    # Get all points in the 2D histogram given a reference value
    #
    def get_points( self, th2 , effref):
        nbinsy = th2.GetNbinsY()
        x = list(); y = list(); errors = list()
        for by in range(nbinsy):
            xproj = th2.ProjectionX('xproj'+str(time.time()),by+1,by+1)
            discr, error = self.find_threshold(xproj,effref)
            dbin = xproj.FindBin(discr)
            x.append(discr); y.append(th2.GetYaxis().GetBinCenter(by+1))
            errors.append( error )
        return x,y,errors



    #
    # Calculate the linear fit given a 2D histogram and reference value and return the slope and offset
    #
    def fit(self, th2,effref):
        x_points, y_points, error_points = self.get_points(th2, effref )
        import array
        from ROOT import TGraphErrors, TF1
        g = TGraphErrors( len(x_points)
                             , array.array('d',y_points,)
                             , array.array('d',x_points)
                             , array.array('d',[0.]*len(x_points))
                             , array.array('d',error_points) )
        firstBinVal = th2.GetYaxis().GetBinLowEdge(th2.GetYaxis().GetFirst())
        lastBinVal = th2.GetYaxis().GetBinLowEdge(th2.GetYaxis().GetLast()+1)
        f1 = TF1('f1','pol1',firstBinVal, lastBinVal)
        g.Fit(f1,"FRq")
        slope = f1.GetParameter(1)
        offset = f1.GetParameter(0)
        return slope, offset, x_points, y_points, error_points



    #
    # Calculate the numerator and denomitator given the 2D histogram and slope/offset parameters
    #
    def calculate_num_and_den_from_hist(self, th2, slope, offset) :

      nbinsy = th2.GetNbinsY()
      th1_num = th2.ProjectionY(th2.GetName()+'_proj'+str(time.time()),1,1)
      th1_num.Reset("ICESM")
      numerator=0; denominator=0
      # Calculate how many events passed by the threshold
      for by in range(nbinsy) :
          xproj = th2.ProjectionX('xproj'+str(time.time()),by+1,by+1)
          # Apply the correction using ax+b formula
          threshold = slope*th2.GetYaxis().GetBinCenter(by+1)+ offset
          dbin = xproj.FindBin(threshold)
          num = xproj.Integral(dbin+1,xproj.GetNbinsX()+1)
          th1_num.SetBinContent(by+1,num)
          numerator+=num
          denominator+=xproj.Integral(-1, xproj.GetNbinsX()+1)

      return numerator/denominator, numerator, denominator


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
        return self.__table


    #
    # Export all models to keras/onnx using the prometheus format
    #
    def export( self, best_sorts, model_output_format , conf_output, reference_name, to_onnx=False,
                max_avgmu = 100, min_avgmu = 0, remove_last=True):


        try:
            os.makedirs('models')
        except:
            pass

        model_etmin_vec = []
        model_etmax_vec = []
        model_etamin_vec = []
        model_etamax_vec = []
        model_paths = []
        slopes = []
        offsets = []


        for et_bin in best_sorts.et_bin.unique():

            for eta_bin in best_sorts.eta_bin.unique():

                row = best_sorts.loc[(best_sorts.et_bin==et_bin) & (best_sorts.eta_bin==eta_bin)]
                if row.shape[0] == 0:
                    continue

                model, _, _, _ = self.__model_generator( row, remove_last=remove_last)


                model_etmin_vec.append( float(self.etbins[et_bin]) )
                model_etmax_vec.append( float(self.etbins[et_bin+1]) )
                model_etamin_vec.append( float(self.etabins[eta_bin]) )
                model_etamax_vec.append( float(self.etabins[eta_bin+1]) )

                model_name = 'models/'+model_output_format%( et_bin, eta_bin )
                model_paths.append( model_name+'.onnx' ) #default is onnx since this will be used into the athena base
                model_json = model.to_json()

                with open(model_name+".json", "w") as json_file:
                    json_file.write(model_json)
                    # saving the model weight separately
                    model.save_weights(model_name+".h5")

                if to_onnx:
                    # NOTE: This is a hack since I am not be able to convert to onnx inside this function. I need to
                    # open a new python instance (call by system) to convert my models.
                    command = 'convert2onnx.py -j {FILE}.json -w {FILE}.h5 -o {FILE}.onnx'.format(FILE=model_name)
                    os.system(command)


                slopes.append( float(row.slope.values[0]) )
                offsets.append( float(row.offset.values[0]) )


        def list_to_str( l ):
            s = str()
            for ll in l:
              s+=str(ll)+'; '
            return s[:-2]

        # Write the config file
        file = TEnv( 'ringer' )
        file.SetValue( "__name__", 'should_be_filled' )
        file.SetValue( "__version__", 'should_be_filled' )
        file.SetValue( "__operation__", reference_name )
        file.SetValue( "__signature__", 'should_be_filled' )
        file.SetValue( "Model__size"  , str(len(model_paths)) )
        file.SetValue( "Model__etmin" , list_to_str(model_etmin_vec) )
        file.SetValue( "Model__etmax" , list_to_str(model_etmax_vec) )
        file.SetValue( "Model__etamin", list_to_str(model_etamin_vec) )
        file.SetValue( "Model__etamax", list_to_str(model_etamax_vec) )
        file.SetValue( "Model__path"  , list_to_str( model_paths ) )
        file.SetValue( "Threshold__size"  , str(len(model_paths)) )
        file.SetValue( "Threshold__etmin" , list_to_str(model_etmin_vec) )
        file.SetValue( "Threshold__etmax" , list_to_str(model_etmax_vec) )
        file.SetValue( "Threshold__etamin", list_to_str(model_etamin_vec) )
        file.SetValue( "Threshold__etamax", list_to_str(model_etamax_vec) )
        file.SetValue( "Threshold__slope" , list_to_str(slopes) )
        file.SetValue( "Threshold__offset", list_to_str(offsets) )
        file.SetValue( "Threshold__MaxAverageMu", list_to_str([max_avgmu]*len(model_paths)))
        file.SetValue( "Threshold__MinAverageMu", list_to_str([min_avgmu]*len(model_paths)))

        MSG_INFO( self, "Export all tuning configuration to %s.", conf_output)
        file.WriteFile(conf_output)

