#!/bin/bash
python main_experiments.py --task_type classification --models deberta-base --epochs 1 --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL && \
python main_experiments.py --task_type classification --models allmpnet --epochs 1 --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL && \
python main_experiments.py --task_type classification --models angle-base --epochs 1 --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL && \
python main_experiments.py --task_type classification --models roberta-base --epochs 1 --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL && \
python main_experiments.py --task_type classification --models bert-base --epochs 1 --nhid 0 --initial_layer 0 --poolings all --agg_layers ALL
