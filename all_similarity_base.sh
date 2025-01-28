#!/bin/bash
python main_experiments.py --task_type similarity --models deberta-base --epochs 0 --batch 0 --kfold 0 --optim adam --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL
python main_experiments.py --task_type similarity --models allmpnet --epochs 0 --batch 0 --kfold 0 --optim adam --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL
python main_experiments.py --task_type similarity --models roberta-base --epochs 0 --batch 0 --kfold 0 --optim adam --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL
python main_experiments.py --task_type similarity --models bert-base --epochs 0 --batch 0 --kfold 0 --optim adam --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL