function[Batches]=GetTestBatch(Source,batch_size,parameter)
    N_batch=ceil(length(Source)/batch_size);
    Batches={};
    for i=1:N_batch
        Begin=batch_size*(i-1)+1;
        End=batch_size*i;
        if End>length(Source)
            End=length(Source);
        end
        current_batch=Batch();
        for j=Begin:End
            source_length=length(Source{j});
            current_batch.SourceLength=[current_batch.SourceLength,source_length];
            if source_length>current_batch.MaxLenSource
                current_batch.MaxLenSource=source_length;
            end
        end
        current_batch.Word=ones(End-Begin+1,current_batch.MaxLenSource);
        Mask=ones(End-Begin+1,current_batch.MaxLenSource);
        for j=Begin:End
            source_length=length(Source{j});
            current_batch.Word(j-Begin+1,current_batch.MaxLenSource-source_length+1:current_batch.MaxLenSource)=Source{j};
            Mask(j-Begin+1,1:current_batch.MaxLenSource-source_length)=0;
        end
        for j=1:current_batch.MaxLenSource
            current_batch.Delete{j}=find(Mask(:,j)==0);
            current_batch.Left{j}=find(Mask(:,j)==1);
        end
        current_batch.Mask=Mask;
        Batches{i}=current_batch;
    end
end
