#!/bin/bash
python main_experiments.py --task_type similarity --models deberta-base --epochs 1 --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL && \
python main_experiments.py --task_type similarity --models allmpnet --epochs 1 --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL && \
python main_experiments.py --task_type similarity --models angle-base --epochs 1 --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL && \
python main_experiments.py --task_type similarity --models roberta-base --epochs 1 --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL && \
python main_experiments.py --task_type similarity --models bert-base --epochs 1 --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL
