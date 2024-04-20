#!/bin/bash

source venv/bin/activate
python -m bp_simunek.scripts.mlda_sample

cp -r $SCRATCH ~/bayes_output