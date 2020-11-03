#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:42:11 2020

@author: kunal001
"""
import networkx as nx
from itertools import combinations
import pathlib
import json
import time
#from multiprocessing import Process 
#from timeout import timeout
import signal
class TimeOutException(Exception):
   pass
 
def alarm_handler(signum, frame):
    print("ALARM signal received")
    raise TimeOutException()
 

input_dirpath=(pathlib.Path(__file__).resolve().parent.parent / 'data'/'MIXER')
output_dirpath=(pathlib.Path(__file__).resolve().parent.parent / 'dataset'/'train')
def get_labels(subgraph):
    label=[]
    _id=0
    node_map={}
    for node, attr in subgraph.nodes(data=True):
        node_map[node]=_id
        _id+=1
        if "inst_type" in attr:
            if 'mos' in attr["inst_type"].lower():
                subgraph.nodes[node]["inst_type"]='MOS'
                label.append('0')
            elif attr["inst_type"].lower() == 'res':
                label.append('1')
            elif 'cap' in attr["inst_type"].lower():
                label.append('2')
            elif attr["inst_type"] == 'net':
                label.append('3')
            else:
                label.append('4')
    return label,node_map
def my_func(g1,g2):
    if g1.size()<g2.size():
        ged=int(nx.graph_edit_distance(g1, g2))
    else:
        ged=int(nx.graph_edit_distance(g2, g1))
    return ged

file2 = open(input_dirpath / 'time.txt',"w+") 

for f1,f2 in combinations(input_dirpath.rglob('*yaml'), 2):
    out_filename = str(f2.stem)+'_'+str(f1.stem)+'.json'
    outpath = output_dirpath / out_filename
    g1=nx.read_yaml(f1)
    g2=nx.read_yaml(f2)
    labels_1,map_1=get_labels(g1)
    labels_2,map_2=get_labels(g2)
    ##calculating GED time
    start = time.time()
    ged=0
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(300)
    try:
        ged=my_func(g1,g2)
    except TimeOutException as ex:
        continue
    signal.alarm(0)
    end = time.time()
    ged= my_func(g1,g2)
    runtime= '\nruntime: '+str(out_filename)+str(len(g1.nodes()))+' and '+str(len(g2.nodes()))+' ged '+str(ged)+' time '+str(end-start)
    print(runtime)
    file2.write(runtime)
    data={'labels_1':labels_1,
            'graph_1':[[map_1[i[0]],map_1[i[1]],i[2]] for i in g1.edges.data('weight',default=1)],
            'labels_2':labels_2,
            'graph_2':[[map_2[i[0]],map_2[i[1]],i[2]] for i in g2.edges.data('weight',default=1)],
            'ged':ged
            }
    
    with open(outpath, 'w') as f:
        json.dump(data, f)
    
file2.close()
