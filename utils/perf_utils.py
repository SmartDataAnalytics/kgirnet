import json

ent_file = open("data/soccer/soccer_entities.txt", "r", encoding="utf-8")

# Get global entities
with open('data/incar/kvret_entities.json') as f:
    global_entity = json.load(f)
    global_entity_list = []
    for key in global_entity.keys():
        if key != 'poi':
            global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
        else:
            for item in global_entity['poi']:
                global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
    global_entity_list = list(set(global_entity_list))

global_entity_list = list(global_entity_list)
# global_entity_list = '_'.join(o))


def processKG(filename):
    kg = ""
    try:
        kg = open('data/KG/clubs/'+filename+".txt", 'r', encoding="utf-8")
    except FileNotFoundError:
        kg = open('data/KG/country/' + filename + ".txt", 'r', encoding="utf-8")

    kg_all = []
    for line in kg:
        kg_all.append(['_'.join(a_line.split(" ")) for a_line in line.strip().split("\t")])
    return kg_all


def get_f1(gold_batch, pred_batch):
    # get f1 scores based on mohammad et. al.
    # print (gold_batch, pred_batch)
    right = 0
    gold_span = gold_batch
    pred_span = pred_batch

    total_en = len(gold_span)
    predicted = len(pred_span)
    for item in pred_span:
        if item in gold_span:
            right += 1
    if predicted == 0:
        precision = 0
    else:
        precision = right / predicted
    if total_en == 0:
        recall = 0
    else:
        recall = right / total_en
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    # fout.flush()
    # fout.close()
    return f1


def get_object(word):
    if '<ent>' in word:
        return word.replace('<ent>', '')
    else:
        return word


def compute_prf(gold, pred, local_kb_word):
    # local_kb_word = [k[0] for k in kb_plain]
    # local_kb_word = [k for k in kb_plain]
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in set(pred):
            if p in global_entity_list or p in local_kb_word:
                if p not in gold:
                    FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    else:
        precision, recall, F1, count = 0, 0, 0, 0
    return F1, count


def compute_f1(gold, pred, localkg):
    # entities = global_entity_list
    epsilon = 0.000000001
    # f1_score = 0.0
    microF1_TRUE = 0.0
    microF1_PRED = 0.0

    for it in range(len(gold)):
        f1_true, count = compute_prf(gold[it], pred[it].split(), localkg[it])
        microF1_TRUE += f1_true
        microF1_PRED += count

    f1_score = microF1_TRUE / float(microF1_PRED + epsilon)
    return f1_score
