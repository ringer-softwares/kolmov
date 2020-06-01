from Gaugi import load, save,Logger, expandFolders
import numpy as np
import sys,os
import glob

__all__ = [
  'add_tag'
]

def add_tag (path, tag):
  task_list = glob.glob(path+'/*')
  print('There are %i tasks to process.' %(len(task_list)))
  for ifolder in task_list:
    print('Processing: %s' %(ifolder.split('/')[-1]))
    job_file_list = expandFolders( ifolder )
    print('This task contains %i job files.' %(len(job_file_list)))
    for f in job_file_list:
      #print(f)
      raw = load( f )
      raw['tag'] = tag
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




