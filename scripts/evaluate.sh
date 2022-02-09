#!/usr/bin/env bash

set -x

config_path=$1
data_path="/workspace/bonito/bonito/data/dna_r9.4.1"
bonito evaluate ${config_path} --directory ${data_path} --batchsize 8 --chunks 1000
