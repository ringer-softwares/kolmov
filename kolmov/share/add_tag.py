from Gaugi import load, save,Logger, expandFolders
import numpy as np
import argparse
import sys,os
import glob

mainLogger = Logger.getModuleLogger("add_tag")
parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()


parser.add_argument('-p','--path', action='store',
        dest='path', required = True,
            help = "The job tuning file.")

parser.add_argument('-t','--tag', action='store',
        dest='tag', required = True, default = '',
            help = "The tuning tag name.")

if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

args = parser.parse_args()

task_list = glob.glob(args.path+'/*')
print('There are %i tasks to process.' %(len(task_list)))
for ifolder in task_list:
  print('Processing: %s' %(ifolder.split('/')[-1]))
  job_file_list = expandFolders( ifolder )
  print('This task contains %i job files.' %(len(job_file_list)))
  for f in job_file_list:
    #print(f)
    raw = load( f )
    raw['tag'] = args.tag
    for ituned in raw['tunedData']:
      for key1 in ituned['history']['reference'].keys():
        iituned = ituned['history']['reference'][key1]
        #print(iituned.keys()) 
        for key2 in iituned.keys():
          if type(iituned[key2]) is tuple:
            x = list(iituned[key2])
            x[1] = int(x[1])
            iituned[key2] = tuple(x)
    # save the altered file
    save( raw, f)




