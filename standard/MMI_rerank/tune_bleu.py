import subprocess,os,numpy;
from tune_length import *

dicfile="../../../data_gr/gr_dict";
file_name="../decode/testing/gr_dev_N_best";
ref_file="../../../data_word_gr/dev_gr";

t_given_s_file="../get_s_given_t/score/t_given_s";
s_given_t_file="../get_s_given_t/score/s_given_t";
index_file="../get_s_given_t/score/index";
length_file="../get_s_given_t/score/length";


read_t_given_s=open(t_given_s_file,"r");
t_given_s=read_t_given_s.readlines();

read_s_given_t=open(s_given_t_file,"r");
s_given_t=read_s_given_t.readlines();

read_index=open(index_file,"r")
indexes=read_index.readlines();

read_length=open(length_file,"r");
lengthes=read_length.readlines();

open_dic_=open(dicfile,"r");
open_dic=open_dic_.readlines();
dic={};
index=0;
for item in open_dic:
    item=item.strip();
    index=index+1;
    dic[str(index)]=item;


read_translation=open(file_name,"r");
translations=read_translation.readlines();

print(len(translations))
print ref_file

MMI=numpy.arange(0.1,0.6,0.1);
G=numpy.arange(-0.02,0,0.005)
candidate_file="towrite";

clear=1;
for mmi in MMI:
    for lambda_ in G:
        print mmi
        print lambda_
        Tune(mmi,lambda_,dic,translations,t_given_s,s_given_t,indexes,lengthes);
        command="./bleu.perl "+ref_file+"<"+candidate_file;
        subprocess.call([command],shell=True);

