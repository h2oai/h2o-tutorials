# download and install h2o here: 
# http://h2o-release.s3.amazonaws.com/h2o/rel-yates/2/index.html
# commands to run:
# > cd ~/h2o
# > java -version
# > wget http://h2o-release.s3.amazonaws.com/h2o/rel-yates/3/h2o-3.24.0.2.zip
# > unzip h2o-3.24.0.2.zip
# > cd h2o-3.24.0.2
# > java -jar h2o.jar
#
# Flow
# to run Flow notebook point to http://10.157.105.198:54321
# 
#
#
# seed to use: 75205
#
# to use H2O from R:

# The following two commands remove any previously installed H2O packages for R.
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o", lib = .libPaths()) }

# Next, we download packages that H2O depends on.
pkgs <- c("RCurl","jsonlite")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}

# Now we download, install and initialize the H2O package for R.
install.packages("h2o", type="source", repos="http://h2o-release.s3.amazonaws.com/h2o/rel-yates/3/R")

# to use H2O from Python:
#
# > pip install requests
# > pip install tabulate
# > pip install "colorama>=0.3.8"
# > pip install future
#
# > pip uninstall h2o
# > pip install http://h2o-release.s3.amazonaws.com/h2o/rel-yates/3/Python/h2o-3.24.0.3-py2.py3-none-any.whl

# Project space:
# cd /Users/gregorykanevsky/Projects/SMU-DataScience-for-Business-H2O-Unit


# Chicago crime dataset (Binomial classification dataset):
# https://s3.amazonaws.com/h2o-public-test-data/smalldata/chicago/chicagoCrimes10k.csv.zip
# name: chicago.crime.hex

# Regression dataset (Wine quality):
# https://s3.amazonaws.com/h2o-public-test-data/smalldata/wine/winequality-redwhite.csv
# name: wine.hex
#
# Source:
# https://data.world/food/wine-quality

# Resources:
# Docs: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html

# Big data dataset - Citibike
# https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/citibike-nyc/2013-10.csv


# Driverless AI docs
# http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/install/aws.html


# Gregory Kanevsky - Sales Engineering
# gregory@h2o.ai
# or sales@h2o.ai