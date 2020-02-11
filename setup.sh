export KOLMOV_PATH=`pwd`

rm -rf .__python__
mkdir .__python__
cd .__python__

ln -s ../Gaugi/python Gaugi
ln -s ../kolmov/python kolmov

export PYTHONPATH=`pwd`:$PYTHONPATH
cd ../