from nltk.translate.meteor_score import single_meteor_score
import numpy as np


files = ['test_predicted_ir_kg_net_soccer_bert.csv','test_predicted_ir_kg_net_incar_bert.csv']
for fl in files:
    sc = []
    with open(fl) as f:
        for i, aline in enumerate(f.readlines()):
            if i:
                info = aline.strip().split("\t")
                sc.append(single_meteor_score(info[1],info[2]))

    print(fl," ::  METEOR score: ",round(np.average(sc)*100,2))