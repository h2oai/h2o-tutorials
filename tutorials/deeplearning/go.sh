#!/bin/bash

## Turn R Markdown into regular Markdown
sed -e 's/```{r}/```r/' deeplearning.Rmd > deeplearning.md

## Turn R Markdown into plain R
sed -n '/```{r}/,/```/p' deeplearning.Rmd | sed '/```/d' > deeplearning.R
