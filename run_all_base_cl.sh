#!/bin/bash
python main_experiments.py --task_type classification --models deberta-base --epochs 1 --batch 1024 --kfold 10 --optim adam --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL