import os,sys;
isDev=int(sys.argv[1]);
if isDev=="1":
    L="dev";
else: L="valid";

if isDev==1:
    target_file="../testing/test/gr_dev_N_best"
    source_file='../../data_gr/dev_en';
else :
    target_file="../testing/test/gr_valid_N_best"
    source_file='../../data_gr/valid_en';

A=open(source_file,"r");
num=0;
G={};
for line in A:
    num=num+1;
    G[num]=line;

B=open(target_file,"r");
A1=open("score/rerank_source_"+L,"w");
A2=open("score/rerank_target_"+L,"w");
A3=open("score/t_given_s_"+L,"w");
A4=open("score/length_"+L,"w");
A5=open("score/index_"+L,"w");

for line in B:
    t1=line.find(" ");
    source_index=int(line[0:t1]);
    t2=line.find(" ",t1+1);
    t_give_s_score=float(line[t1:t2]);
    end_t=line.find("50001");
    target_line=line[t2+1:end_t];

    A1.write(G[source_index]);
    A2.write(target_line+"\n");
    A3.write(str(t_give_s_score)+"\n");
    A4.write(str(len(target_line.split(" ")))+"\n");
    A5.write(str(source_index)+"\n");
    
