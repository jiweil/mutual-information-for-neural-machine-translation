function Get_Score(parameter)
    End=0;
    fd_train_source=fopen(parameter.train_source_file);   %read source
    fd_train_target=fopen(parameter.train_target_file);   %read target
    parameter.train_source_file
    parameter.train_target_file
    index=0;
    sum_score=0;
    sum_num=0;
    while 1
        index=index+1;
        [batch,End]=ReadTrainData(fd_train_source,fd_train_target,parameter);   %transform data to batches
        %not the end of document
        [lstm,h_t,c]=Forward(batch,parameter,0,0);
        %LSTM Forward
        score(h_t,batch,parameter,parameter.write_filename);      
%         sum_score=sum_score+Score;
%         sum_num=sum_num+num;
        %softmax
        if End==1
            break;
        end
    end
end


function[current_batch,End]=ReadTrainData(fd_s,fd_t,parameter)
    tline_s = fgets(fd_s);
    tline_t = fgets(fd_t);
    i=0;
    Source={};Target={};
    End=0;
    while ischar(tline_s)
        i=i+1;
        text_s=deblank(tline_s);
        text_t=deblank(tline_t);
        if parameter.Source_Target_Same_Language~=1
            Source{i}=wrev(str2num(text_s))+parameter.TargetVocab;  
            %reverse inputs
        else
            Source{i}=str2num(text_s);    
        end
        Target{i}=[str2num(text_t),parameter.stop];     
        Target{i}=[str2num(text_t)]; 
        %add document_end_token
        if i==parameter.batch_size
            break;
        end
        tline_s = fgets(fd_s);
        tline_t = fgets(fd_t);
    end

    if ischar(tline_s)==0
        End=1;
    end

    current_batch=Batch();
    N=length(Source);
    for j=1:N
        source_length=length(Source{j});
        current_batch.SourceLength=[current_batch.SourceLength,source_length];
        if source_length>current_batch.MaxLenSource
            current_batch.MaxLenSource=source_length;
        end
        target_length=length(Target{j});
        if target_length>current_batch.MaxLenTarget
            current_batch.MaxLenTarget=target_length;
        end
    end
    total_length=current_batch.MaxLenSource+current_batch.MaxLenTarget;
    current_batch.MaxLen=total_length;
    current_batch.Word=ones(N,total_length);
    Mask=ones(N,total_length);
    % Mask: labeling positions where no words exisit. The purpose is to work on sentences in bulk making program faster
    for j=1:N
        source_length=length(Source{j});
        target_length=length(Target{j});
        current_batch.Word(j,current_batch.MaxLenSource-source_length+1:current_batch.MaxLenSource)=Source{j};      
        %words within sentences 
        current_batch.Word(j,current_batch.MaxLenSource+1:current_batch.MaxLenSource+target_length)=Target{j};
        Mask(j,1:current_batch.MaxLenSource-source_length)=0;       
        Mask(j,current_batch.MaxLenSource+target_length+1:end)=0;   
        % label positions without tokens 0
        current_batch.N_word=current_batch.N_word+target_length;
    end
    for j=1:total_length
        current_batch.Delete{j}=find(Mask(:,j)==0);
        current_batch.Left{j}=find(Mask(:,j)==1);
    end
    current_batch.Mask=Mask;
end
