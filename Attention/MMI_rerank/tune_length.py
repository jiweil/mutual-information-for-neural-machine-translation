def Tune(MMI,Lambda,Dic,translations,t_given_s,s_given_t,indexes,lengthes):
    File=open("towrite","w")
    line_index=1;
    max_score=-10000;
    best_translation ="";

    count=0;
    for i in range(0,len(translations)):
        index=int(indexes[i]);
        if index!=line_index:
            t1=best_translation.find(" ");
            t2=best_translation.find(" ",t1+1);
            t3=best_translation.find(" 50001");
            word_list=best_translation[t2:t3].strip().split(" ");
            for word in word_list:
                File.write(Dic[word]+" ");
            File.write("\n")
            line_index=index;
            max_score=-10000;
            best_translation ="";
        current_score=(1-MMI)*float(t_given_s[i])+MMI*float(s_given_t[i])+Lambda*int(lengthes[i]);

        if current_score>max_score:
            max_score=current_score;
            best_translation=translations[i];

    t1=best_translation.find(" ");
    t2=best_translation.find(" ",t1+1);
    t3=best_translation.find(" 50001");
    word_list=best_translation[t2:t3].strip().split(" ");

    word_list=best_translation[0:t1].strip().split(" ");
    for word in word_list:
        File.write(Dic[word]+" ");

    File.write("\n")
    File.close()
