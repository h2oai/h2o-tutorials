#!/bin/bash

sed -e '1,\%```r%s:^:#:;/^```/,\%```r%s:^:#:;/```/d' README.md > ensembles-stacking.R
