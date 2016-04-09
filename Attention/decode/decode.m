function[]=decode()
addpath('../../misc');
Test_source_file{1}='../../data_gr/valid_en';
Test_source_file{2}='../../data_gr/dev_en';
save_fil{1}='test/gr_valid_N_best';
save_fil{2}='test/gr_dev_N_best';
load '../training/gr_given_en0.2/12.mat'


for i=1:2
    parameter.batch_size=1;
    parameter.test_source_file=Test_source_file{i};
    parameter.save_file=save_fil{i};
    
    Test=ReadTestData(parameter.test_source_file,parameter);
    TestBatches=GetTestBatch(Test,parameter.batch_size,parameter);

    parameter.multi=10;
    decode_beam_attention(parameter,TestBatches,parameter.save_file);

end

end

function[Source]=ReadTestData(source_file,parameter)
    fd_s=fopen(source_file);
    tline_s = fgets(fd_s);
    i=0;
    Source={};
    End=0;
    while ischar(tline_s)
        i=i+1;
        text_s=deblank(tline_s);
        Source{i}=wrev(str2num(text_s))+parameter.TargetVocab;  
            %reverse inputs
        %add document_end_token
        tline_s = fgets(fd_s);
    end
end


function[Batches]=GetTestBatch(Source,batch_size,parameter)
    N_batch=ceil(length(Source)/batch_size);
    Batches={};
    for i=1:N_batch
        Begin=batch_size*(i-1)+1;
        End=batch_size*i;
        if End>length(Source)
            End=length(Source);
        end
        current_batch.SourceLength=0;
        current_batch.MaxLenSource=0;
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

