function[]=decode_beam_standard(parameter,TestBatches,filename)
    disp('decode')
    filename1=strcat(filename,'N_best');
    parameter.beamSize=200;
    for batch_index=1:length(TestBatches)
    %for batch_index=1:1000
        batch_index
        tic
        batch=TestBatches{batch_index};
        max_length=ceil(batch.MaxLenSource*1.5);
        Word=batch.Word;
        N=size(Word,1);
        SourceLength=batch.SourceLength;
        [lstms,last_h_t,last_c_t]=Forward(batch,parameter,0,1);
        zeroState=zeroMatrix([parameter.hidden,parameter.beamSize]);
        
        [first_scores,first_words]=BeamStep(parameter,last_h_t{parameter.layer_num},1,[],1);
%         first_words
%         first_scores
        beamHistory=oneMatrix([parameter.beamSize,max_length]);
        
        beamHistory(:,1)=first_words(:);
        beamScores=first_scores(:)';
        
        beamStates=cell(parameter.layer_num,1);
        for ll=1:parameter.layer_num
            beamStates{ll}.c_t=repmat(last_c_t{ll},1,parameter.beamSize);
            beamStates{ll}.h_t=repmat(last_h_t{ll},1,parameter.beamSize);
        end
        H=[];
        Store_Scores=[];
        Num_Word=[];
        for position=1:max_length
            words=beamHistory(:,position);
            for ll=1:parameter.layer_num
                if ll == 1
                    x_t=parameter.vect(:,words);
                else
                    x_t=beamStates{ll-1}.h_t;
                end
                h_t_1 = beamStates{ll}.h_t;
                c_t_1 = beamStates{ll}.c_t;
                [beamStates{ll}, h_t, c_t]=lstmUnit(parameter.W_T{ll},parameter,x_t,h_t_1, c_t_1, ll, -1,0);
                beamStates{ll}.h_t = h_t;
                beamStates{ll}.c_t = c_t;
            end
            
            [all_next_scores,all_next_words]=BeamStep(parameter,beamStates{parameter.layer_num}.h_t,0,words,position+1);
            all_next_scores=bsxfun(@plus,all_next_scores,beamScores);

                all_next_scores=reshape(all_next_scores,[size(all_next_scores,1)*size(all_next_scores,2),1]);
                all_next_words=reshape(all_next_words,[size(all_next_scores,1)*size(all_next_scores,2),1]);
                [all_next_scores,sorted_Indices]=sort(all_next_scores,'descend');
            
                sorted_next_words=all_next_words(sorted_Indices);

                end_index=find(sorted_next_words==parameter.stop);
                
                if length(end_index)~=0 && position>batch.MaxLenSource*0.5
%                 && position+1>=parameter.index
                   % end_index=end_index(1);
                    previous_index=floor((sorted_Indices(end_index)-1)/parameter.beamSize)+1;
                    beamHistory(previous_index,position+1)=parameter.stop;
                    H=[H;beamHistory(previous_index,:)];
                    Store_Scores=[Store_Scores;all_next_scores(end_index)/(position+1)];
                    Num_Word=[Num_Word;repmat(position+1,length(end_index),1)];
                end
                next_word_index=find(sorted_next_words~=parameter.stop,parameter.beamSize);
                beamScores=all_next_scores(next_word_index)';

                previous_index=floor((sorted_Indices(next_word_index)-1)/parameter.beamSize)+1;
                beamHistory(1:parameter.beamSize,:)=beamHistory(previous_index,:);
                beamHistory(1:parameter.beamSize,position+1)=sorted_next_words(next_word_index);       
                
                %if length(find(sorted_next_words(next_word_index)==1))~=0
                %    sorted_next_words(next_word_index)
                %end
                %if length(find(beamHistory(:,1:position+1)==1))~=0
                %    disp('his')
                %end
                
                for ll=1:parameter.layer_num
                    beamStates{ll}.c_t=beamStates{ll}.c_t(:,previous_index);
                    beamStates{ll}.h_t=beamStates{ll}.h_t(:,previous_index);
                end
                [beamScores,rank]=sort(beamScores,'descend');
                beamHistory=beamHistory(rank,:);
                
                for ll=1:parameter.layer_num
                    beamStates{ll}.c_t=beamStates{ll}.c_t(:,rank);
                    beamStates{ll}.h_t=beamStates{ll}.h_t(:,rank);
                end
            if position+1==max_length
                break;
            end

        end
        
         [A1,A2]=sort(Store_Scores,'descend');
         Store_Scores=Store_Scores(A2);
         
         H=H(A2,:);
         if length(H)~=0
             write=0;
             N1=size(H,1);
             H=[repmat(batch_index,N1,1),Store_Scores,H];
             dlmwrite(filename,H,'delimiter',' ','-append');
        else
            H=[batch_index,0,beamHistory(1,:),parameter.stop];
            dlmwrite(filename,H,'delimiter',' ','-append');
        end
         
         toc
%          for senId=1:length(lang_Scores)
%             vector=H(senId,:);
%             %vector=vector(1:floor(2*batch.SourceLength(1)));
%             stop_sign=find(vector==parameter.stop);
%             v=vector(1:stop_sign-1);
%             v_length=length(v);
% 
%             if v_length<10
%                 continue;
%             end
% %             if length(v)<10
% %                 continue;
% %             end
%             
%             dlmwrite(filename,v,'delimiter',' ','-append');
%             write=1;
%             break;
%          end
%          if write==0
%              if length(vector)==0
%                  dlmwrite(filename,[2,16,13,32,3],'delimiter',' ','-append');
%              else
%                 vector=H(1,:);
%                 stop_sign=find(vector==parameter.stop);
%                 v=vector(1:stop_sign-1);
%                 dlmwrite(filename,v,'delimiter',' ','-append');
%              end
%          end
    end
end


function[logP]=BeamStepSequence(parameter,h_t,isFirst)
    soft_W=parameter.soft_W;
    if isFirst==1 scores=soft_W(1:end,:)*h_t;
    else scores=soft_W(1:end,:)*h_t; 
    end
    mx = max(scores);
    scores = bsxfun(@minus, scores, mx);
    logP=bsxfun(@minus, scores, log(sum(exp(scores))));
end


function[select_logP,select_words]=BeamStep(parameter,h_t,isFirst,pre_word,index)
    scroes1=BeamStepSequence(parameter,h_t,isFirst);
    logP=scroes1;
    if isFirst==1
        logP(50001,:)=-Inf;
    end
    %logP(1,:)=-Inf;
    [sortedLogP, sortedWords]=sort(logP, 'descend');    
    select_words=sortedWords(1:parameter.beamSize, :);
    select_logP=sortedLogP(1:parameter.beamSize, :);
end
