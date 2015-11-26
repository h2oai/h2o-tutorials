#!/bin/bash

## Turn R Markdown into regular Markdown
sed -e 's/```{r.*}/```r/' deeplearning.Rmd > deeplearning.md
cp deeplearning.md README.md

## Turn R Markdown into plain R
sed -e '1,\%```{r.*}%s:^:#:;/^```/,\%```{r.*}%s:^:#:;/```/d' deeplearning.Rmd > deeplearning.R
