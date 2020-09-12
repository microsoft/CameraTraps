#!/bin/sh
# Script for filtering the tf output 
# Usage: bash filter_tf_output.sh train_serengeti_inception_v4.sh
bash $* 3>&1 1>&2 2>&3 3>&- | egrep "^INFO" 
