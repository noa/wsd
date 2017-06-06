#! /usr/bin/env bash

EXE='python wsd.py --mode topk'
VOCAB='data/vocab.100000.txt'
INPUT='example_topk_input.txt'
MODEL='checkpoints/model.ckpt-100000'

$EXE --vocab_path $VOCAB --input_path $INPUT --checkpoint_path $MODEL
