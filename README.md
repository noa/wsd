# wsd

The main entry point is `wsd.py` which has three modes: `train`,
`eval`, and `topk`. See the documentation for the command line flags
in `wsd.py` for options. To download a sample pre-trained model, run
the `download_model.sh` script. For an example call to generate top K
results, see the `topk.sh` script (which assumes `download_model.sh`
has been called in the root directory).
