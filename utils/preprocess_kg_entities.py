import os
import json

def save2file(dataset, entities):
    with open("data/"+dataset+"_entities.txt","w",encoding="utf-8") as f:
        mapping_file = "data/"+dataset+"_entitymapping.json"
        mapping = dict()
        for i, entity in enumerate(entities):
            f.write(entity+"\n")
            mapping[entity] = i

        json.dump(mapping, open(mapping_file, "w"), indent=4)

def getEntities_fromKG(kgtype, kgname):
    sub_n_obj = set()
    with open("data/KG/"+kgtype+"/"+kgname, "r", encoding="utf-8") as f:
        triples = f.readlines()
        for triple in triples:
            s, _, o = triple.split("\t")
            sub_n_obj.add(s.lower())
            sub_n_obj.add(o.strip().lower())
    return sub_n_obj


def generate_Entities(dataset):
    entities = set()
    if dataset == "soccer":
        kgtype = ["clubs","country"]
        for kgs in kgtype:
            for kg in os.listdir("data/KG/"+kgs):
                entities = entities.union(getEntities_fromKG(kgs, kg))
    else:
        kgs = "incar"
        for kg in os.listdir("data/KG/"+kgs):
            entities = entities.union(getEntities_fromKG(kgs, kg))
    save2file(dataset, list(entities))


if __name__ == "__main__":

    generate_Entities("soccer")
    print("Generated Soccer Entities!!")
    generate_Entities("incar")
    print("Generated Soccer Entities!!")