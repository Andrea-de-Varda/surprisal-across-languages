from transformers import XGLMTokenizer, XGLMForCausalLM
import torch
from scipy.special import softmax
import torch.nn.functional as F
import numpy as np
from os import chdir
import pandas as pd
from tqdm import tqdm
import wordfreq
import pickle
import re

# (XGLM) https://huggingface.co/facebook/xglm-564M
# -- no Hebrew, Dutch, norwegian
# -- https://arxiv.org/abs/2112.10668

chdir("/path/to/your/wd/")

toker = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")

def pickle_save(file, filename):
    print("Saving to", filename)
    with open(filename, 'wb') as f:
        pickle.dump(file, f)

def get_surprisal(prompt):
    inputs = toker(prompt, return_tensors="pt")
    input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]
    outputs = model(**inputs, labels=input_ids)
    logits = outputs.logits
    logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2))
    return [-item[0] for item in logprobs.tolist()[0]]

def tok_maker(a):
    # Credit to Ben S. https://stackoverflow.com/questions/74458282/match-strings-of-different-length-in-two-lists-of-different-length
    plainseq = " ".join(a)
    b = [re.sub("▁", "", item) for item in toker.tokenize(plainseq)]
    c = []
    for element in a:
        temp_list = []
        while "".join(temp_list) != element:
            temp_list.append(b.pop(0))
        c.append(temp_list)
    return c

def get_surprisal_tokens(tokens, unique):
    s = get_surprisal(" ".join(tokens))
    toks = tok_maker(tokens)
    theindex = 0
    out = {}
    out_ismulti = {}
    for index, word in enumerate(toks):
        if len(word) == 1:
            surp = s[theindex]
            theindex += 1
            out[unique[index]] = surp
            out_ismulti[unique[index]] = 0
        else:
            surp = s[theindex:theindex+len(word)]
            theindex += len(word)
            out[unique[index]] = sum(surp)
            out_ismulti[unique[index]] = 1
    return out, out_ismulti

########
# MECO #
########

meco = pd.read_csv("/path/to/your/Meco_l1.csv", index_col=None)
meco = meco.iloc[: , 1:]; print("N (with NA) =", len(meco))
meco = meco[meco['ia'].notna()]; print("N (no NA) =",len(meco))
meco["trialid"] = meco.trialid.astype(int)
meco["itemid"] = meco.itemid.astype(int)
meco["ianum"] = meco.ianum.astype(int)
meco["unique"] = meco.trialid.astype(str)+"-"+meco.ianum.astype(str)+"-"+meco.lang
meco = meco[~meco["subid"].isin(["ee_09", "ee_22", "ru_8"])] # wrong itemid assignment (as in 1.2 release)

languages = ['it', 'ee', 'en', 'fi', 'ge', 'gr', 'he', 'du', 'ko', 'no', 'ru', 'sp', 'tr']
out = {lang:[] for lang in languages}
for language in languages: 
    print("\n\n", language.upper())
    for textid in set(meco.trialid):
        print(textid)
        temp = meco[(meco["lang"] == language) & (meco["trialid"] == textid)]
        therange = range(1, max(temp.ianum)+1)
        temp_out = []
        temp_out_unique = []
        for itemindex in therange:
            try:
                token = list(set(temp[temp.ianum == itemindex].ia))
                unique = list(set(temp[temp.ianum == itemindex].unique))
                if len(token) > 1:
                    print(textid, itemindex, token)
                temp_out_unique.append(unique[0])
                temp_out.append(token[0])
            except IndexError:
                print(language, textid, itemindex)
        out[language].append([textid, temp_out, temp_out_unique])

def get_d_surp(lang, exclude=None):
    d_out_s = {}
    d_out_multi = {}
    for index, text in enumerate(out[lang]):
        if index+1 in exclude:
            pass
        else:
            print("Processing text", index+1)
            thetext = text[1]
            unique = text[2]
            d_s, d_multi = get_surprisal_tokens(thetext, unique)
            d_out_s = {**d_out_s, **d_s}
            d_out_multi = {**d_out_multi, **d_multi}
    return d_out_s, d_out_multi
        
it_s, it_m = get_d_surp("it")
ee_s, ee_m = get_d_surp("ee")
en_s, en_m = get_d_surp("en")
fi_s, fi_m = get_d_surp("fi")
ge_s, ge_m = get_d_surp("ge")
gr_s, gr_m = get_d_surp("gr")
ko_s, ko_m = get_d_surp("ko", exclude=[10])
ru_s, ru_m = get_d_surp("ru")
sp_s, sp_m = get_d_surp("sp")
tr_s, tr_m = get_d_surp("tr")

s = {**it_s, **ee_s, **en_s, **fi_s, **ge_s, **gr_s, **gr_s, **ko_s, **ru_s, **sp_s, **tr_s}
m = {**it_m, **ee_m, **en_m, **fi_m, **ge_m, **gr_m, **gr_m, **ko_m, **ru_m, **sp_m, **tr_m}

meco["s"] = meco["unique"].map(s)
meco["m"] = meco["unique"].map(m)

##############################
# adding frequency estimates #
# no wordfreq for Estonian   #
##############################

import plainstream

estonian = pd.read_csv("estonian_freq.csv", sep="\t")
tot_ee = sum(estonian.Total) # 820241
freq_ee = {}
for index, row in estonian.iterrows():
    freq_ee[row.Word] = np.log10(row.Total)

freq = []
for index, row in meco.iterrows():
    w = row.ia
    l = row.lang
    if l == "ee":
        try:
            f = freq_ee[w]
            freq.append(f)
        except KeyError:
            freq.append(1)
    else:
        if l == "du": # need to change lang codes
            l = "nl"
        if l == "ge":
            l = "de"
        if l == "gr":
            l = "el"
        if l == "no":
            l = "nb"
        if l == "sp":
            l = "es"
        try:
            f = wordfreq.zipf_frequency(w, l)
            freq.append(f)
        except KeyError:
            freq.append(1)
    if index%100==0:
        print(round(index/len(meco), 4))

meco["freq"] = freq

meco.to_csv("meco_xglm")    
