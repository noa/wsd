#! /usr/bin/env bash

DIR="models"

if [ ! -d $DIR ]; then
    mkdir $DIR
fi

wget -nc -P $DIR https://storage.googleapis.com/spoken_wsd_models/ptb_model/model.ckpt-100000.data-00000-of-00001
wget -nc -P $DIR https://storage.googleapis.com/spoken_wsd_models/ptb_model/model.ckpt-100000.index
wget -nc -P $DIR https://storage.googleapis.com/spoken_wsd_models/ptb_model/model.ckpt-100000.meta
wget -nc -P $DIR https://storage.googleapis.com/spoken_wsd_models/ptb_model/vocab.100000.txt
