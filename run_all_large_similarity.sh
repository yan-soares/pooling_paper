#!/bin/bash
python main_experiments.py --task_type similarity --models deberta-large --epochs 1 --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL && \
python main_experiments.py --task_type similarity --models angle-large --epochs 1 --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL && \
python main_experiments.py --task_type similarity --models roberta-large --epochs 1 --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL && \
python main_experiments.py --task_type similarity --models bert-large --epochs 1 --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL
