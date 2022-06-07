#!/bin/sh

#python3 evaluate.py --ctx-length 5 10 15 20 25 30 50 --ds-type transcripts --gpu 0
python3 evaluate.py --ctx-length 5 10 15 20 25 30 50 --ds-type transcripts --case-sensitive --gpu 0
#python3 evaluate.py --ctx-length 5 10 15 20 25 30 50 --case-sensitive --ds-type aligned --gpu 0
#python3 evaluate.py --ctx-length 5 10 15 20 25 30 50 --fill-na unk --ds-type aligned --gpu 0
# python3 evaluate.py --ctx-length 5 10 15 20 25 30 50 --fill-na replace --ds-type aligned --gpu 0