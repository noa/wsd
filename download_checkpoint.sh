#! /usr/bin/env bash

if [ ! -d "checkpoints" ]; then
    mkdir checkpoints
fi

wget -P checkpoints https://storage.googleapis.com/spoken_wsd_models/ptb_model/model.ckpt-100000.data-00000-of-00001
wget -P checkpoints https://storage.googleapis.com/spoken_wsd_models/ptb_model/model.ckpt-100000.index
wget -P checkpoints https://storage.googleapis.com/spoken_wsd_models/ptb_model/model.ckpt-100000.meta
