import csv,json,os





def readconv(tsvfile):
    allconv = []
    with open(tsvfile) as f:
        alllines = f.readlines()
        for i,aline in enumerate(alllines):
            allconv.append(aline.strip().split(","))
    return allconv


def save2file():

    datasets = ["train","test","val"]

    for d in datasets:
        tsvf = "data/incar/"+d+"_fixed.csv"
        allconv = readconv(tsvf)
        savedir = "data/incar/manually_annotated/"+d+"_sketch/"
        runningfile = allconv[0][0]
        savelist = []
        counter=0
        for aline in allconv:
            fname,q,a,input_rel,corr_rel,obj,input_ent = aline[0],aline[1],aline[2],aline[3],aline[4],aline[5],aline[6]
            counter+=1
            if fname!=runningfile:
                json.dump(savelist,open(savedir+runningfile,"w",encoding="utf-8"),indent=4)  #saving
                #resetting everyting for new (conversation) json file
                runningfile=fname
                savelist = []
                counter=1
            if d=="train":
                temp = {
                    "q"+str(counter): q,
                    "a"+str(counter): a,
                    "input_rel"+str(counter): input_rel,
                    "corr_rel"+str(counter): corr_rel,
                    "obj"+str(counter):obj,
                    "input_ent"+str(counter): input_ent
                }
            else:
                temp = {
                    "q"+str(counter): q,
                    "a"+str(counter): a,
                    "input_rel"+str(counter): "",
                    "corr_rel"+str(counter): "",
                    "obj"+str(counter):obj,
                    "input_ent"+str(counter): ""
                }
            savelist.append(temp)

        #for the last file
        savelist.append(temp)
        json.dump(savelist, open(savedir + runningfile, "w", encoding="utf-8"), indent=4)
        print("DONE !!")


save2file()
