#!/bin/bash

## Turn R Markdown into regular Markdown
echo "##AUTO-GENERATED - DO NOT EDIT##" > deeplearning.md
sed -e 's/```{r.*}/```r/' deeplearning.Rmd >> deeplearning.md
cp deeplearning.md README.md

## Turn R Markdown into plain R
echo "##AUTO-GENERATED - DO NOT EDIT##" > deeplearning.R
sed -e '1,\%```{r.*}%s:^:#:;/^```/,\%```{r.*}%s:^:#:;/```/d' deeplearning.Rmd >> deeplearning.R
