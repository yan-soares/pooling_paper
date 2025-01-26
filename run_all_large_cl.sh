#!/bin/bash
python main_experiments.py --task_type classification --models deberta-large --epochs 1 --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL && \
python main_experiments.py --task_type classification --models angle-large --epochs 1 --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL && \
python main_experiments.py --task_type classification --models roberta-large --epochs 1 --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL && \
python main_experiments.py --task_type classification --models bert-large --epochs 1 --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL
