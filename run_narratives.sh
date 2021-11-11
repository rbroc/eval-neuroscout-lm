#!/bin/sh

python3 validate_narratives.py --ctx-length 5 10 15 20 25 30 50 100 --ds-type transcripts --gpu 0
python3 validate_narratives.py --ctx-length 5 10 15 20 25 30 50 100 --case-sensitive --ds-type aligned --gpu 0
python3 validate_narratives.py --ctx-length 5 10 15 20 25 30 50 100 --fill-na unk --ds-type aligned --gpu 0
python3 validate_narratives.py --ctx-length 5 10 15 20 25 30 50 100 --fill-na replace --ds-type aligned --gpu 0