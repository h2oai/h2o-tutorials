#!/bin/bash

#
# Note:  This run script is meant to be run inside the docker container.
#

# set -x
set -e

if [ "x$1" != "x" ]; then
    d=$1
    cd $d
    shift
    exec "$@"
fi

logdir=/log/`date "+%Y%m%d-%H%M%S"`
mkdir -p "$logdir"

echo "-------------------------"
echo "Welcome to H2O World 2017"
echo "-------------------------"
echo ""
echo "- Connect to Jupyter notebook on port 8888 (password: h2o)"
echo "- Connect to RStudio on port 8787 (username/password: h2o/h2o)"

source /home/h2o/Miniconda3/bin/activate h2o

(cd /home/h2o && \
 jupyter --paths >> "$logdir"/jupyter.log && \
 nohup jupyter notebook --ip='*' --no-browser --allow-root >> "$logdir"/jupyter.log 2>&1 &)

(cd /home/h2o && \
 sudo rstudio-server start >> "$logdir"/rstudio-server.log)

(cd /home/h2o/h2o-3.16.0.2 && \
 sudo java -jar h2o.jar >> "$logdir"/h2o.log)

# 10 years
sleep 3650d
