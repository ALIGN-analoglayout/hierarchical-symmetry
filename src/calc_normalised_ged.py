#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:42:11 2020

@author: kunal001
"""
import networkx as nx
import pathlib
import json
 

input_dirpath=(pathlib.Path(__file__).resolve().parent.parent / 'dataset'/'train')
for f1 in input_dirpath.rglob('*json'):
    with open(f1, 'r') as f:
        existing_val=json.load(f)
        ged=existing_val['ged']
        nodes=len(existing_val['labels_1'])+len(existing_val['labels_2'])+len(existing_val['graph_1'])+len(existing_val['graph_2'])
        val=ged/nodes
        print(f1, val, nodes, ged )
input_dirpath=(pathlib.Path(__file__).resolve().parent.parent / 'dataset'/'test')
for f1 in input_dirpath.rglob('*json'):
    with open(f1, 'r') as f:
        existing_val=json.load(f)
        ged=existing_val['ged']
        nodes=len(existing_val['labels_1'])+len(existing_val['labels_2'])+len(existing_val['graph_1'])+len(existing_val['graph_2'])
        val=ged/nodes
        print(f1, val, nodes, ged )
