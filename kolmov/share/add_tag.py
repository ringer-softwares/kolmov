







from Gaugi import load, save,Logger, expandFolders
import numpy as np
import argparse
import sys,os


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


for f in expandFolders( args.path ):
  print(f)

  raw = load( f )

  raw['tag'] = args.tag
    
  for ituned in raw['tunedData']:
    for key1 in ituned['history']['reference'].keys():
        
      iituned = ituned['history']['reference'][key1]
      print(iituned.keys()) 
      for key2 in iituned.keys():
          if type(iituned[key2]) is tuple:
              x = list(iituned[key2])
              x[1] = int(x[1])
              print(x)
              iituned[key2] = tuple(x)

  save( raw, f)




