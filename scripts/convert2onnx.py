#!/usr/bin/env python3


import argparse
parser = argparse.ArgumentParser(description = '', add_help = False)
parser.add_argument('-j', action='store',
    dest='json_file', required = True,
    help = "keras json file.")
parser.add_argument('-w', action='store',
    dest='h5_file', required = True,
    help = "keras h5 file.")
parser.add_argument('-o', action='store',
    dest='output_file', required = True,
    help = "onnx output file.")
import sys,os
if len(sys.argv)==1:
  	parser.print_help()
  	sys.exit(1)
args = parser.parse_args()


import keras2onnx, onnx
from tensorflow.keras.models import model_from_json
model = model_from_json( open( args.json_file ).read() )
model.load_weights( args.h5_file )
onnx_model = keras2onnx.convert_keras(model)
print('Saving ONNX file as '+ args.output_file)
onnx.save_model( onnx_model, args.output_file )

