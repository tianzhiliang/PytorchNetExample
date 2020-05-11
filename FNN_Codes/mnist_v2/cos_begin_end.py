import sys,os,math,random
import torch
import torch.nn as nn

def digital_format_on_list(x, len=6):
    return [round(a, len) for a in x]

def load(file):
    model_dict = torch.load(file)
    print("model_dict:", model_dict)

    if 'fc1weight.weight' in model_dict:
        data = model_dict['fc1weight.weight']
    elif 'fc1.weight' in model_dict:
        data = model_dict['fc1.weight']
    #return data
    return data.cpu().tolist()[0]

def print_data(data, outfile):
    f = open(outfile, "w")
    #data = ori_data.cpu().tolist()
    for d in data:
        f.write(str(d) + "\n")

def cos_sim(alist, blist):
    print("alist:", alist)
    print("blist:", blist)
    alist = list(map(float, alist))
    blist = list(map(float, blist))
    aa, bb, ab = 0, 0, 0
    for a, b in zip(alist, blist):
        aa += a*a
        bb += b*b
        ab += a*b

    aabb = math.sqrt(aa * bb)
    if aabb == 0:
        return 0

    return ab / aabb

def main():
    file_begin = sys.argv[1]
    file_end = sys.argv[2]
    data_begin = load(file_begin)
    data_end = load(file_end)
    cos = cos_sim(data_begin, data_end)
    print("cos:", cos)
    print_data(data_begin, file_begin + ".weight")
    print_data(data_end, file_end + ".weight")

main()
