import pandas as pd
from collections import defaultdict
import numpy as np
import json

annotated_files = '/home/debanjan/submission_soccer/data/incar/conversations/manual_anno/'
out_directory = '/home/debanjan/submission_soccer/data/incar/conversations/output_anno/'


def read_files(dataset):
    dat = np.array(pd.read_csv(annotated_files+dataset+'_incar.tsv', sep='\t'))
    data_dict = defaultdict(list)
    for d in dat:
        data_dict[d[0]].append(d[1:])

    for k, v in data_dict.items():
        # item_dict = defaultdict()
        item_list = []
        for j, item in enumerate(v):
            convo_dict = {}
            convo_dict['q'+str(j+1)] = item[0]
            convo_dict['a'+str(j+1)] = item[1]
            kvr_items = item[2].replace('[', '').replace(']', '').replace("'",'').split(',')
            convo_dict['kvr' + str(j + 1)] = [item.strip() for item in kvr_items]
            try:
                convo_dict['object'+str(j+1)] = item[3].split(',')
            except AttributeError:
                convo_dict['object' + str(j + 1)] = item[3]
            convo_dict['type'+str(j+1)] = item[4]
            item_list.append(convo_dict)
        # item_dict[k] = item_list
        with open(out_directory+dataset+'/'+k+'.json', 'w') as f:
            json.dump(item_list, f, indent=4)


datasets = ['train', 'val', 'test']
for d in datasets:
    read_files(d)






