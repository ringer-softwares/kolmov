
#
# DISCLAIMER: You should have root installed at your system to use it.
#

__all__ = ['fit_table']

from Gaugi.messenger import Logger, LoggingLevel
from Gaugi.messenger.macros import *
from Gaugi.tex import *
from Gaugi.monet.AtlasStyle import *
from Gaugi.monet.PlotFunctions import *
from Gaugi.monet.TAxisFunctions import *
from Gaugi import progressbar
from copy import copy
from itertools import product

import numpy as np
import pandas
import time
import collections
import os


from ROOT import gROOT, kTRUE
gROOT.SetBatch(kTRUE)
import ROOT
from ROOT import kBird,kBlackBody
ROOT.gErrorIgnoreLevel=ROOT.kFatal

# allow gpu growth 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#
# Linear correction class table
#
class fit_table(Logger):

    #
    # Constructor
    #
    def __init__(self, generator, etbins, etabins, x_bin_size, y_bin_size, ymin, ymax,
                 false_alarm_limit=0.5,
                 level=LoggingLevel.INFO,
                 xmin_percentage=1,
                 xmax_percentage=99,
                 plot_stage='Internal',
                 palette=kBlackBody,
                 xmin=None,
                 xmax=None):

        # init base class
        Logger.__init__(self, level=level)
        self.__generator = generator
        self.__etbins = etbins
        self.__etabins = etabins
        self.__ymin = ymin
        self.__ymax = ymax
        self.__x_bin_size = x_bin_size
        self.__y_bin_size = y_bin_size
        self.__false_alarm_limit = false_alarm_limit
        self.__xmin_percentage=xmin_percentage
        self.__xmax_percentage=xmax_percentage
        self.__plot_stage=plot_stage
        self.__xmin=xmin
        self.__xmax=xmax
        self.__palette=palette


    #
    # Fill correction table
    #
    def fill( self, data_paths,  models, reference_values, output_dir,
              verbose=False, except_these_bins = [] ):

        from Gaugi.monet.AtlasStyle import SetAtlasStyle
        SetAtlasStyle()
        # create directory
        localpath = os.getcwd()+'/'+output_dir
        try:
            if not os.path.exists(localpath): os.makedirs(localpath)
        except:
            MSG_WARNING( self,'The director %s exist.', localpath)



        # make template dataframe
        dataframe = collections.OrderedDict({
                      'name':[],
                      'et_bin':[],
                      'eta_bin':[],
                      'reference_signal_passed':[],
                      'reference_signal_total':[],
                      'reference_signal_eff':[],
                      'reference_background_passed':[],
                      'reference_background_total':[],
                      'reference_background_eff':[],
                      'signal_passed':[],
                      'signal_total':[],
                      'signal_eff':[],
                      'background_passed':[],
                      'background_total':[],
                      'background_eff':[],
                      'signal_corrected_passed':[],
                      'signal_corrected_total':[],
                      'signal_corrected_eff':[],
                      'background_corrected_passed':[],
                      'background_corrected_total':[],
                      'background_corrected_eff':[],
                     })

        # reduce verbose
        def add(key,value):
          dataframe[key].append(value)


        # Loop over all et/eta bins
        for et_bin, eta_bin in progressbar(product(range(len(self.__etbins)-1),range(len(self.__etabins)-1)),
                                           (len(self.__etbins)-1)*(len(self.__etabins)-1), prefix = "Fitting... " ):
            path = data_paths[et_bin][eta_bin]
            data, target, avgmu = self.__generator(path)
            references = reference_values[et_bin][eta_bin]

            model = models[et_bin][eta_bin]
            model['thresholds'] = {}

            # Get the predictions
            outputs = model['model'].predict(data, batch_size=1024, verbose=verbose).flatten()

            # Get all limits using the output
            xmin = self.__xmin if self.__xmin else int(np.percentile(outputs , self.__xmin_percentage))
            xmax = self.__xmax if self.__xmax else int(np.percentile(outputs, self.__xmax_percentage))

            MSG_DEBUG(self, 'Setting xmin to %1.2f and xmax to %1.2f', xmin, xmax)
            xbins = int((xmax-xmin)/self.__x_bin_size)
            ybins = int((self.__ymax-self.__ymin)/self.__y_bin_size)

            # Fill 2D histograms
            from ROOT import TH2F
            import array
            th2_signal = TH2F( 'th2_signal_et%d_eta%d'%(et_bin,eta_bin), '', xbins, xmin, xmax, ybins, self.__ymin, self.__ymax )
            w = array.array( 'd', np.ones_like( outputs[target==1] ) )
            th2_signal.FillN( len(outputs[target==1]), array.array('d',  outputs[target==1].tolist()),  array.array('d',avgmu[target==1].tolist()), w)
            th2_background = TH2F( 'th2_background_et%d_eta%d'%(et_bin,eta_bin), '', xbins,xmin, xmax, ybins, self.__ymin, self.__ymax )
            w = array.array( 'd', np.ones_like( outputs[target==0] ) )
            th2_background.FillN( len(outputs[target==0]), array.array('d',outputs[target!=1].tolist()), array.array('d',avgmu[target!=1].tolist()), w)

            MSG_DEBUG( self, 'Applying linear correction to et%d_eta%d bin.', et_bin, eta_bin)

            for name, ref in references.items():
                
                if ref['pd_epsilon'] == 0.0:
                    ref_value = ref['pd']
                else:
                    add_fac = (1-ref['pd'])*ref['pd_epsilon']
                    ref_value = ref['pd'] + add_fac
                    MSG_INFO(self, 'Add %1.2f %% in reference pd -> new reference pd: %1.2f', ref['pd_epsilon'], add_fac)

                false_alarm = 1.0
                while false_alarm > self.__false_alarm_limit:

                    # Get the threshold when we not apply any linear correction
                    threshold, _ = self.find_threshold( th2_signal.ProjectionX(), ref_value )

                    # Get the efficiency without linear adjustment
                    #signal_eff, signal_num, signal_den = self.calculate_num_and_den_from_hist(th2_signal, 0.0, threshold)
                    signal_eff, signal_num, signal_den = self.calculate_num_and_den_from_output(outputs[target==1], avgmu[target==1], 0.0, threshold)
                    #background_eff, background_num, background_den = self.calculate_num_and_den_from_hist(th2_background, 0.0, threshold)
                    background_eff, background_num, background_den = self.calculate_num_and_den_from_output(outputs[target!=1], avgmu[target!=1], 0.0, threshold)

                    # Apply the linear adjustment and fix it in case of positive slope
                    slope, offset, x_points, y_points, error_points = self.fit( th2_signal, ref_value )
                    
                    # put inside of the ref array
                    apply_fit = True

                    # case 1: The user select each bin will not be corrected
                    for (this_et_bin, this_eta_bin) in except_these_bins:
                        if et_bin == this_et_bin and eta_bin == this_eta_bin:
                            apply_fit = False
                    # case 2: positive slope
                    if slope > 0:
                        MSG_WARNING(self, "Retrieved positive angular factor of the linear correction, setting to 0!")
                        apply_fit = False


                    slope = slope if apply_fit else 0
                    offset = offset if apply_fit else threshold






                    # Get the efficiency with linear adjustment
                    #signal_corrected_eff, signal_corrected_num, signal_corrected_den = self.calculate_num_and_den_from_hist(th2_signal, slope, offset)
                    signal_corrected_eff, signal_corrected_num, signal_corrected_den = self.calculate_num_and_den_from_output(outputs[target==1], \
                                                                                                                              avgmu[target==1], slope, offset)
                    #background_corrected_eff, background_corrected_num, background_corrected_den = self.calculate_num_and_den_from_hits(th2_background, slope, offset)
                    background_corrected_eff, background_corrected_num, background_corrected_den = self.calculate_num_and_den_from_output(outputs[target!=1], \
                                                                                                                                          avgmu[target!=1], slope, offset)

                    false_alarm = background_corrected_num/background_corrected_den # get the passed/total

                    if false_alarm > self.__false_alarm_limit:
                        # Reduce the reference value by hand
                        ref_value-=0.0025

                MSG_DEBUG( self, 'Reference name: %s, target: %1.2f%%', name, ref['pd']*100 )
                MSG_DEBUG( self, 'Signal with correction is: %1.2f%%', signal_corrected_num/signal_corrected_den * 100 )
                MSG_DEBUG( self, 'Background with correction is: %1.2f%%', background_corrected_num/background_corrected_den * 100 )

                # decore the model array
                model['thresholds'][name] = {'offset':offset,
                                             'slope':slope,
                                             'threshold' : threshold,
                                             'reference_pd': ref['pd'],
                                             'reference_fa':ref['fa'],
                                             }
                paths = []

                # prepate 2D histograms
                info = models[et_bin][eta_bin]['thresholds'][name]
                outname = localpath+'/th2_signal_%s_et%d_eta%d'%(name,et_bin,eta_bin)
                output = self.plot_2d_hist( th2_signal, slope, offset, x_points, y_points, error_points, outname, xlabel='<#mu>',
                                   etBinIdx=et_bin, etaBinIdx=eta_bin, etBins=self.__etbins,etaBins=self.__etabins,
                                   plot_stage=self.__plot_stage)
                paths.append(output)
                outname = localpath+'/th2_background_%s_et%d_eta%d'%(name,et_bin,eta_bin)
                output = self.plot_2d_hist( th2_background, slope, offset, x_points, y_points, error_points, outname, xlabel='<#mu>',
                                   etBinIdx=et_bin, etaBinIdx=eta_bin, etBins=self.__etbins,etaBins=self.__etabins,
                                   plot_stage=self.__plot_stage)
                paths.append(output)

                model['thresholds'][name]['figures'] = paths



                # et/eta bin information
                add( 'name'                        , name )
                add( 'et_bin'                      , et_bin  )
                add( 'eta_bin'                     , eta_bin )
                # reference values
                add( 'reference_signal_passed'     , int(ref['pd']*signal_den) )
                add( 'reference_signal_total'      , signal_den )
                add( 'reference_signal_eff'        , ref['pd'] )
                add( 'reference_background_passed' , int(ref['fa']*background_den) )
                add( 'reference_background_total'  , background_den )
                add( 'reference_background_eff'    , ref['fa'] )
                # non-corrected values
                add( 'signal_passed'               , signal_num )
                add( 'signal_total'                , signal_den )
                add( 'signal_eff'                  , signal_num/signal_den )
                add( 'background_passed'           , background_num )
                add( 'background_total'            , background_den )
                add( 'background_eff'              , background_num/background_den )
                # corrected values
                add( 'signal_corrected_passed'     , signal_corrected_num )
                add( 'signal_corrected_total'      , signal_corrected_den )
                add( 'signal_corrected_eff'        , signal_corrected_num/signal_corrected_den )
                add( 'background_corrected_passed' , background_corrected_num )
                add( 'background_corrected_total'  , background_corrected_den )
                add( 'background_corrected_eff'    , background_corrected_num/background_corrected_den )

        # convert to pandas dataframe
        self.__table = pandas.DataFrame( dataframe )


    #
    # Get the pandas table
    #
    def table(self):
        return self.__table


    #
    # Dump bearmer report
    #
    def dump_beamer_table( self, table, models, title, output_pdf ):


        # Slide maker
        with BeamerTexReportTemplate1( theme = 'Berlin'
                                     , _toPDF = True
                                     , title = title
                                     , outputFile = output_pdf
                                     , font = 'structurebold' ):

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


            for name in table.name.unique():

                with BeamerSection( name = name.replace('_','\_') ):

                    # prepare figures
                    with BeamerSubSection( name = 'Correction plots for each phase space'):

                        for etBinIdx in range( len(self.__etbins)-1 ):

                            for etaBinIdx in range( len(self.__etabins)-1 ):

                                current_table = table.loc[ (table.name==name) & (table.et_bin==etBinIdx) & (table.eta_bin==etaBinIdx)]

                                # prepate 2D histograms
                                paths = models[etBinIdx][etaBinIdx]['thresholds'][name]['figures']

                                title = 'Energy between %s in %s (et%d\_eta%d)'%(etbins_str[etBinIdx],
                                                                                 etabins_str[etaBinIdx],
                                                                                 etBinIdx,etaBinIdx)

                                BeamerMultiFigureSlide( title = title
                                , paths = paths
                                , nDivWidth = 2 # x
                                , nDivHeight = 1 # y
                                , texts=None
                                , fortran = False
                                , usedHeight = 0.6
                                , usedWidth = 1.
                                )





                    # prepate table
                    with BeamerSubSection( name = 'Efficiency Values' ):

                        # Prepare phase space table
                        lines1 = []
                        lines1 += [ HLine(_contextManaged = False) ]
                        lines1 += [ HLine(_contextManaged = False) ]
                        lines1 += [ TableLine( columns = [''] + [s for s in etabins_str], _contextManaged = False ) ]
                        lines1 += [ HLine(_contextManaged = False) ]

                        for etBinIdx in range( len(self.__etbins)-1 ):

                            values_det = []; values_fa = []
                            for etaBinIdx in range( len(self.__etabins)-1 ):
                                # Get the current bin table
                                current_table = table.loc[ (table.name==name) & (table.et_bin==etBinIdx) & (table.eta_bin==etaBinIdx)]
                                det = current_table.signal_corrected_eff.values[0] * 100
                                fa = current_table.background_corrected_eff.values[0] * 100

                                # Get reference pd
                                ref = current_table.reference_signal_eff.values[0] * 100

                                if (det-ref) > 0.0:
                                    values_det.append( ('\\cellcolor[HTML]{9AFF99}%1.2f ($\\uparrow$%1.2f[$\\Delta_{ref}$])')%(det,det-ref) )
                                elif (det-ref) < 0.0:
                                    values_det.append( ('\\cellcolor[HTML]{F28871}%1.2f ($\\downarrow$%1.2f[$\\Delta_{ref}$])')%(det,det-ref) )
                                else:
                                    values_det.append( ('\\cellcolor[HTML]{9AFF99}%1.2f')%(det) )

                                ref = current_table.reference_background_eff.values[0] * 100

                                factor = fa/ref if ref else 0.
                                if (fa-ref) > 0.0:
                                    values_fa.append( ('\\cellcolor[HTML]{F28871}%1.2f ($\\rightarrow$%1.2f$\\times\\text{FR}_{ref}$)')%(fa,factor) )
                                elif (fa-ref) < 0.0:
                                    values_fa.append( ('\\cellcolor[HTML]{9AFF99}%1.2f ($\\rightarrow$%1.2f$\\times\\text{FR}_{ref}$)')%(fa,factor) )
                                else:
                                    values_fa.append( ('\\cellcolor[HTML]{9AFF99}%1.2f')%(fa) )

                            lines1 += [ TableLine( columns = [etbins_str[etBinIdx]] + values_det   , _contextManaged = False ) ]
                            lines1 += [ TableLine( columns = [''] + values_fa , _contextManaged = False ) ]
                            lines1 += [ HLine(_contextManaged = False) ]
                        lines1 += [ HLine(_contextManaged = False) ]


                        # Prepare integrated table
                        lines2 = []
                        lines2 += [ HLine(_contextManaged = False) ]
                        lines2 += [ HLine(_contextManaged = False) ]
                        lines2 += [ TableLine( columns = ['',r'$P_{D}[\%]$',r'$F_{a}[\%]$'], _contextManaged = False ) ]
                        lines2 += [ HLine(_contextManaged = False) ]

                        # Get all et/eta bins
                        current_table = table.loc[table.name==name]

                        # Get reference values
                        passed_fa = current_table.reference_background_passed.sum()
                        total_fa = current_table.reference_background_total.sum()
                        fa = passed_fa/total_fa * 100
                        passed_det = current_table.reference_signal_passed.sum()
                        total_det = current_table.reference_signal_total.sum()
                        det = passed_det/total_det * 100

                        lines2 += [ TableLine( columns = ['Reference','%1.2f (%d/%d)'%(det,passed_det,total_det),
                                              '%1.2f (%d/%d)'%(fa,passed_fa,total_fa)]  , _contextManaged = False ) ]


                        # Get corrected values
                        passed_fa = current_table.background_corrected_passed.sum()
                        total_fa = current_table.background_corrected_total.sum()
                        fa = passed_fa/total_fa * 100
                        passed_det = current_table.signal_corrected_passed.sum()
                        total_det = current_table.signal_corrected_total.sum()
                        det = passed_det/total_det * 100

                        lines2 += [ TableLine( columns = [name.replace('_','\_'),'%1.2f (%d/%d)'%(det,passed_det,total_det),
                                              '%1.2f (%d/%d)'%(fa,passed_fa,total_fa)]  , _contextManaged = False ) ]

                        # Get non-corrected values
                        passed_fa = current_table.background_passed.sum()
                        total_fa = current_table.background_total.sum()
                        fa = passed_fa/total_fa * 100
                        passed_det = current_table.signal_passed.sum()
                        total_det = current_table.signal_total.sum()
                        det = passed_det/total_det * 100

                        lines2 += [ TableLine( columns = [name.replace('_','\_')+' (not corrected)','%1.2f (%d/%d)'%(det,passed_det,total_det),
                                                '%1.2f (%d/%d)'%(fa,passed_fa,total_fa)]  , _contextManaged = False ) ]

                        lines2 += [ HLine(_contextManaged = False) ]
                        lines2 += [ HLine(_contextManaged = False) ]

                        with BeamerSlide( title = "Efficiency Values After Correction"  ):
                            with Table( caption = '$P_{d}$ and $F_{a}$ for all phase space regions.') as _table:
                                with ResizeBox( size = 1 ) as rb:
                                    with Tabular( columns = 'l' + 'c' * len(etabins_str) ) as tabular:
                                        tabular = tabular
                                        for line in lines1:
                                            if isinstance(line, TableLine):
                                                tabular += line
                                            else:
                                                TableLine(line, rounding = None)
                            with Table( caption = 'Integrated efficiency comparison.') as _table:
                                with ResizeBox( size = 0.6 ) as rb:
                                    with Tabular( columns = 'l' + 'c' * 3 ) as tabular:
                                        tabular = tabular
                                        for line in lines2:
                                            if isinstance(line, TableLine):
                                                tabular += line
                                            else:
                                                TableLine(line, rounding = None)




    #
    # Export all models to keras/onnx using the prometheus format
    #
    def export( self, models, model_output_format , conf_output, reference_name, to_onnx=False):

        from ROOT import TEnv

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

        # serialize all models
        for etBinIdx in range( len(self.__etbins)-1 ):
            for etaBinIdx in range( len(self.__etabins)-1 ):

                model = models[etBinIdx][etaBinIdx]

                if model['etBinIdx']!=etBinIdx:
                    MSG_FATAL(self, 'Model etBinIdx (%d) is different than etBinIdx (%d). Abort.', model['etBinIdx'], etBinIdx)

                if model['etaBinIdx']!=etaBinIdx:
                    MSG_FATAL(self, 'Model etaBinIdx (%d) is different than etaBinIdx (%d). Abort.', model['etaBinIdx'], etaBinIdx)

                model_etmin_vec.append( model['etBin'][0] )
                model_etmax_vec.append( model['etBin'][1] )
                model_etamin_vec.append( model['etaBin'][0] )
                model_etamax_vec.append( model['etaBin'][1] )
                etBinIdx = model['etBinIdx']
                etaBinIdx = model['etaBinIdx']

                model_name = 'models/'+model_output_format%( etBinIdx, etaBinIdx )
                model_paths.append( model_name+'.onnx' ) #default is onnx since this will be used into the athena base
                model_json = model['model'].to_json()
                with open(model_name+".json", "w") as json_file:
                    json_file.write(model_json)
                    # saving the model weight separately
                    model['model'].save_weights(model_name+".h5")

                if to_onnx:
                    # NOTE: This is a hack since I am not be able to convert to onnx inside this function. I need to
                    # open a new python instance (call by system) to convert my models.
                    command = 'convert2onnx.py -j {FILE}.json -w {FILE}.h5 -o {FILE}.onnx'.format(FILE=model_name)
                    os.system(command)


                slopes.append( model['thresholds'][reference_name]['slope'] )
                offsets.append( model['thresholds'][reference_name]['offset'] )


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
        file.SetValue( "Threshold__MaxAverageMu", list_to_str([100]*len(model_paths)))
        MSG_INFO( self, "Export all tuning configuration to %s.", conf_output)
        file.WriteFile(conf_output)




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

      # Calculate the efficiency histogram
      th1_den = th2.ProjectionY(th2.GetName()+'_proj'+str(time.time()),1,1)
      th1_eff = th1_num.Clone()
      th1_eff.Divide(th1_den)
      # Fix the error bar
      for bx in range(th1_eff.GetNbinsX()):
          if th1_den.GetBinContent(bx+1) != 0 :
              eff = th1_eff.GetBinContent(bx+1)
              try:
                  error = np.sqrt(eff*(1-eff)/th1_den.GetBinContent(bx+1))
              except:
                  error=0
              th1_eff.SetBinError(bx+1,eff)
          else:
              th1_eff.SetBinError(bx+1,0)

      return th1_eff, numerator, denominator


    def calculate_num_and_den_from_output(self, output, avgmu, slope, offset) :
      thr = avgmu*slope + offset
      denominator = len(output)
      numerator = sum( output>thr  )
      return denominator/numerator, numerator, denominator



    #
    # Plot 2D histogram function based on ROOT used by fit_table class
    #
    def plot_2d_hist( self, th2, slope, offset, x_points, y_points, error_points, outname, xlabel='',
                      etBinIdx=None, etaBinIdx=None, etBins=None,etaBins=None, plot_stage='Internal'):

        from ROOT import TCanvas, gStyle, TLegend, gPad, TLatex, kAzure, kRed, kBlue, kBlack,TLine,kBird, kOrange,kGray
        from ROOT import TGraphErrors,TF1,TColor
        import array

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

        # Create canvas and add 2D histogram
        #gStyle.SetPalette(kBird) # default
        gStyle.SetPalette(self.__palette)
        canvas = TCanvas('canvas','canvas',500, 500)
        canvas.SetRightMargin(0.15)
        canvas.SetTopMargin(0.15)
        th2.GetXaxis().SetTitle('Neural Network output (Discriminant)')
        th2.GetYaxis().SetTitle(xlabel)
        th2.GetZaxis().SetTitle('Count')
        th2.Draw('colz')
        canvas.SetLogz()


        # Add dots and line
        g = TGraphErrors(len(x_points), array.array('d',x_points), array.array('d',y_points), array.array('d',error_points), array.array('d',[0]*len(x_points)))
        g.SetMarkerColor(kBlue)
        g.SetMarkerStyle(8)
        g.SetMarkerSize(1)
        g.Draw("P same")
        line = TLine(slope*th2.GetYaxis().GetXmin()+offset,th2.GetYaxis().GetXmin(), slope*th2.GetYaxis().GetXmax()+offset, th2.GetYaxis().GetXmax())
        line.SetLineColor(kBlack)
        line.SetLineWidth(2)
        line.Draw()

        # Add text labels into the canvas
        AddATLASLabel(canvas, 0.16,0.94,plot_stage)
        text = toStrBin(etlist=etBins, etidx=etBinIdx)
        text+= ', '+toStrBin(etalist=etaBins, etaidx=etaBinIdx)
        AddTexLabel(canvas, 0.16, 0.885, text, textsize=.035)

        # Format and save
        FormatCanvasAxes(canvas, XLabelSize=16, YLabelSize=16, XTitleOffset=0.87, ZLabelSize=16,ZTitleSize=16, YTitleOffset=0.87, ZTitleOffset=1.1)
        SetAxisLabels(canvas,'Neural Network output (Discriminant)',xlabel)
        canvas.SaveAs(outname+'.pdf')
        canvas.SaveAs(outname+'.C')
        return outname+'.pdf'



#
# Quick test to dev.
#
if __name__ == "__main__":

    from kolmov import crossval_table

    def create_op_dict(op):
        d = {
                  op+'_pd_ref'    : "reference/"+op+"_cutbased/pd_ref#0",
                  op+'_fa_ref'    : "reference/"+op+"_cutbased/fa_ref#0",
                  op+'_sp_ref'    : "reference/"+op+"_cutbased/sp_ref",
                  op+'_pd_val'    : "reference/"+op+"_cutbased/pd_val#0",
                  op+'_fa_val'    : "reference/"+op+"_cutbased/fa_val#0",
                  op+'_sp_val'    : "reference/"+op+"_cutbased/sp_val",
                  op+'_pd_op'     : "reference/"+op+"_cutbased/pd_op#0",
                  op+'_fa_op'     : "reference/"+op+"_cutbased/fa_op#0",
                  op+'_sp_op'     : "reference/"+op+"_cutbased/sp_op",

                  # Counts
                  op+'_pd_ref_passed'    : "reference/"+op+"_cutbased/pd_ref#1",
                  op+'_fa_ref_passed'    : "reference/"+op+"_cutbased/fa_ref#1",
                  op+'_pd_ref_total'     : "reference/"+op+"_cutbased/pd_ref#2",
                  op+'_fa_ref_total'     : "reference/"+op+"_cutbased/fa_ref#2",
                  op+'_pd_val_passed'    : "reference/"+op+"_cutbased/pd_val#1",
                  op+'_fa_val_passed'    : "reference/"+op+"_cutbased/fa_val#1",
                  op+'_pd_val_total'     : "reference/"+op+"_cutbased/pd_val#2",
                  op+'_fa_val_total'     : "reference/"+op+"_cutbased/fa_val#2",
                  op+'_pd_op_passed'     : "reference/"+op+"_cutbased/pd_op#1",
                  op+'_fa_op_passed'     : "reference/"+op+"_cutbased/fa_op#1",
                  op+'_pd_op_total'      : "reference/"+op+"_cutbased/pd_op#2",
                  op+'_fa_op_total'      : "reference/"+op+"_cutbased/fa_op#2",
        }
        return d


    tuned_info = collections.OrderedDict( {
                  # validation
                  "max_sp_val"      : 'summary/max_sp_val',
                  "max_sp_pd_val"   : 'summary/max_sp_pd_val#0',
                  "max_sp_fa_val"   : 'summary/max_sp_fa_val#0',
                  # Operation
                  "max_sp_op"       : 'summary/max_sp_op',
                  "max_sp_pd_op"    : 'summary/max_sp_pd_op#0',
                  "max_sp_fa_op"    : 'summary/max_sp_fa_op#0',
                  } )

    tuned_info.update(create_op_dict('tight'))
    tuned_info.update(create_op_dict('medium'))
    tuned_info.update(create_op_dict('loose'))
    tuned_info.update(create_op_dict('vloose'))


    etbins = [15,20,30,40,50,100000]
    etabins = [0, 0.8 , 1.37, 1.54, 2.37, 2.5]

    cv  = crossval_table( tuned_info, etbins = etbins , etabins = etabins )
    #cv.fill( '/Volumes/castor/tuning_data/Zee/v10/*.r2/*/*.gz', 'v10')
    #cv.fill( '/home/jodafons/public/tunings/v10/*.r2/*/*.gz', 'v10')
    #cv.to_csv( 'v10.csv' )
    cv.from_csv( 'v10.csv' )
    best_inits = cv.filter_inits("max_sp_val")
    best_inits = best_inits.loc[(best_inits.model_idx==0)]
    best_sorts = cv.filter_sorts(best_inits, 'max_sp_val')
    best_models = cv.get_best_models(best_sorts, remove_last=True)



    #
    # Generator to read, prepare data and get all references
    #
    def generator( path ):

        def norm1( data ):
            norms = np.abs( data.sum(axis=1) )
            norms[norms==0] = 1
            return data/norms[:,None]
        from Gaugi import load
        import numpy as np
        d = load(path)
        feature_names = d['features'].tolist()
        data = norm1(d['data'][:,1:101])
        target = d['target']
        avgmu = d['data'][:,0]
        references = ['T0HLTElectronT2CaloTight','T0HLTElectronT2CaloMedium','T0HLTElectronT2CaloLoose','T0HLTElectronT2CaloVLoose']
        ref_dict = {}
        for ref in references:
            answers = d['data'][:, feature_names.index(ref)]
            signal_passed = sum(answers[target==1])
            signal_total = len(answers[target==1])
            background_passed = sum(answers[target==0])
            background_total = len(answers[target==0])
            pd = signal_passed/signal_total
            fa = background_passed/background_total
            ref_dict[ref] = {'signal_passed': signal_passed, 'signal_total': signal_total, 'pd' : pd,
                             'background_passed': background_passed, 'background_total': background_total, 'fa': fa}

        return data, target, avgmu



    path = '/Volumes/castor/cern_data/files/Zee/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et{ET}_eta{ETA}.npz'
    ref_path = '/Volumes/castor/cern_data/files/Zee/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97/references/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et{ET}_eta{ETA}.ref.pic.gz'

    #path = '~/public/cern_data/files/Zee/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97/data17_13TeV.AllPeriods.sgn.probes_lhmedium_EGAM1.bkg.VProbes_EGAM7.GRL_v97_et{ET}_eta{ETA}.npz'
    paths = [[ path.format(ET=et,ETA=eta) for eta in range(5)] for et in range(5)]
    ref_paths = [[ ref_path.format(ET=et,ETA=eta) for eta in range(5)] for et in range(5)]
    ref_matrix = [[ {} for eta in range(5)] for et in range(5)]
    references = ['T0HLTElectronT2CaloTight','T0HLTElectronT2CaloMedium','T0HLTElectronT2CaloLoose','T0HLTElectronT2CaloVLoose']
    references = ['tight_cutbased', 'medium_cutbased' , 'loose_cutbased', 'vloose_cutbased']
    from saphyra.core import ReferenceReader
    for et_bin in range(5):
        for eta_bin in range(5):
            for name in references:
                refObj = ReferenceReader().load(ref_paths[et_bin][eta_bin])
                pd = refObj.getSgnPassed(name)/refObj.getSgnTotal(name)
                fa = refObj.getBkgPassed(name)/refObj.getBkgTotal(name)
                ref_matrix[et_bin][eta_bin][name] = {'pd':pd, 'fa':fa}




    # get best models
    etbins = [15,20]
    etabins = [0, 0.8]
    ct  = fit_table( generator, etbins , etabins, 0.02, 1.5, 16, 60 )
    ct.fill(paths, best_models, ref_matrix, 'test_dir')


    table = ct.table()
    ct.dump_beamer_table(table, best_models, 'test', 'test')

    ct.export(best_models, 'model_et%d_eta%d', 'config_tight.conf', 'tight_cutbased', to_onnx=True)



