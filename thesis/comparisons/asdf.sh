#/bin/bash

#cd plotfinal/; python plot.py paris-mse.csv paris-mse; python plot.py paris-loss.csv paris-loss; cd ..
cmd1="cd plotfinal/; python plot.py $1-mse.csv $1-mse; python plot.py $1-loss.csv $1-loss; cd .."
cd plotfinal/
python plot.py $1-mse.csv pdf/$1-mse
python plot.py $1-loss.csv pdf/$1-loss
cd ..
