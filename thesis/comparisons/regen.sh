#!/bin/bash

./asdf.sh paris
./asdf.sh paris-nobatchnorm
./asdf.sh dan-dense
./asdf.sh curro
cd plotfinal
python crossplot.py mse -6
python crossplot.py mse -3
python crossplot.py mse 0
python crossplot.py mse 3
python crossplot.py mse 6
python crossplot.py loss -6
python crossplot.py loss -3
python crossplot.py loss 0
python crossplot.py loss 3
python crossplot.py loss 6
cd ..
