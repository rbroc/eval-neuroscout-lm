#!/bin/sh

python3 validate_narratives.py --ctx-length 10 --ds-type transcripts --gpu 0
python3 validate_narratives.py --ctx-length 10 --case-sensitive --ds-type aligned --gpu 0
python3 validate_narratives.py --ctx-length 10 --fill-na unk --ds-type aligned --gpu 0
python3 validate_narratives.py --ctx-length 10 --fill-na replace --ds-type aligned --gpu 0