#!/bin/bash


python generate_plots.py oxides-verification-PBE-v1 'FLEUR@LAPW+LO'
python generate_plots.py unaries-verification-PBE-v1 'FLEUR@LAPW+LO'

for i in epsilon V0_rel_diff B1_rel_diff B0_rel_diff; do

python generate_histos.py oxides-verification-PBE-v1 $i 'FLEUR@LAPW+LO'
python generate_histos.py unaries-verification-PBE-v1 $i 'FLEUR@LAPW+LO'
done
