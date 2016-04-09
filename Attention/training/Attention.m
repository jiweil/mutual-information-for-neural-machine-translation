function[]=Attention(targetIsEn)
addpath('../../misc');
n= gpuDeviceCount;

gpu_index=1
gpuDevice(gpu_index);

parameter.dimension=1000;
parameter.alpha=1;
parameter.fix_alpha=parameter.alpha;

parameter.layer_num=4;  %number of layer
parameter.hidden=1000;
parameter.lstm_out_tanh=0;
parameter.Initial=0.1;
parameter.dropout=0.2;  %drop-out rate
parameter.isTraining=1;
parameter.CheckGrad=0;  %whether check gradient or not.
parameter.PreTrainEmb=0;    %whether using pre-trained embeddings
parameter.update_embedding=1;   %whether update word embeddings
parameter.batch_size=128;    %mini-batch size
%whether source and target is of the same language. For author-encoder task, it is.
parameter.maxGradNorm=5;    %gradient clipping
parameter.clip=0;
parameter.reverse=1;
parameter.lr=5;
parameter.read=0;

if parameter.CheckGrad==1&&parameter.dropout~=0  %use identical dropout-vector for gradient checking
    parameter.drop_left=randSimpleMatrix([parameter.hidden,1])<1-parameter.dropout;
end

parameter.nonlinear_gate_f = @sigmoid;
parameter.nonlinear_gate_f_prime = @sigmoidPrime;
parameter.nonlinear_f = @tanh;
parameter.nonlinear_f_prime = @tanhPrime;

parameter.French=0;
parameter.Target_en=targetIsEn;
if parameter.Target_en==1
    parameter.save_folder='en_given_gr0.2/';
    train_source_file='../../data_gr/train_gr';
    train_target_file='../../data_gr/train_en';
    dev_source_file='../../data_gr/dev_gr';
    dev_target_file='../../data_gr/dev_en';
    test_source_file='../../data_gr/valid_gr';
    test_target_file='../../data_gr/valid_en';
    N_line=4468840;
    parameter.SourceVocab=50000;
    parameter.TargetVocab=50001;
    parameter.start_half_iter=8;
end
if parameter.Target_en==0
    parameter.save_folder='gr_given_en0.2/';
    train_source_file='../../data_gr/en';
    train_target_file='../../data_gr/gr';
    dev_source_file='../../data_gr/dev_en';
    dev_target_file='../../data_gr/dev_gr';
    test_source_file='../../data_gr/valid_en';
    test_target_file='../../data_gr/valid_gr';
    N_line=4468840;
    parameter.SourceVocab=50000;
    parameter.TargetVocab=50001;
    parameter.start_half_iter=8;
end

parameter.Vocab=parameter.SourceVocab+parameter.TargetVocab;
parameter.stop=parameter.TargetVocab;

parameter
iter=0;

if parameter.read==1
    disp('read');
    parameter.read_file=strcat(parameter.save_folder,'/',num2str(iter));
    parameter=ReadParameter(parameter);     %read from exisitng parameter
    testing(test_source_file,test_target_file,parameter);

else [parameter]=Initial(parameter);        %rand initialization
end
parameter

disp('begin')

while 1
    iter=iter+1
    if iter>parameter.start_half_iter
        disp('start cutting lr,  current lr');
        parameter.alpha=parameter.alpha*0.5;
        disp(parameter.alpha);
    end
    End=0;
    fd_train_source=fopen(train_source_file);   %read source
    fd_train_target=fopen(train_target_file);   %read target
    sum_cost=0;
    sum_num=0;
    batch_n=0;
    tic
    while 1 
        batch_n=batch_n+1;
        [batch,End]=ReadTrainData(fd_train_source,fd_train_target,parameter,0);   %transform data to batches
        if size(batch.Word,2)>90
            continue;
        end
        %size(batch.Word)
        if End~=1 || (End==1&& length(batch.Word)~=0)   
            %not the end of document
            [source_h,lstm,h_t,c]=Forward(batch,parameter,1,0);   
            %LSTM Forward
            [batch_cost,grad]=softmax(h_t,batch,parameter);      
            %disp(1/exp(-batch_cost/batch.N_word))
            %softmax
            if (isnan(batch_cost)||isinf(batch_cost)) &&End~=1  
            %if gradient explodes
                if parameter.clip==1
                    fprintf('die !! Hopeless!!\n');
                    %load parameters stores from last step, and skip lately used batches
                    disp(batch_n)
                else parameter.clip=1;
                end
                if End==1 break;    
                %end of documents
                else continue;
                end
            end
            if parameter.isTraining==1
                grad=Backward(source_h,h_t,batch,grad,parameter,lstm,c);  
                clear h_t;
                clear source_h;
                if 1==parameter.CheckGrad
                    check(grad,batch,parameter)
                end
                %backward propagation
                [parameter]=update_parameter(parameter,grad);   
                %update parameter
                clear grad;
                clear lstm;
                clear c;
            end
        end
        if End==1
            fclose(fd_train_source);
            fclose(fd_train_target);
            break;
        end
        if mod(batch_n,20000)==0
            batch_n
            testing(test_source_file,test_target_file,parameter);
        end
    end
    disp('testing')
    testing(test_source_file,test_target_file,parameter);
    SaveParameter(parameter,iter,parameter.save_folder);  
    
    toc
    %save parameter
end
end


function[]=testing(test_source_file,test_target_file,parameter)
    fd_test_source=fopen(test_source_file);   %read source
    fd_test_target=fopen(test_target_file);   %read target
    cost=0;
    num_word=0;
    while 1 
        [batch,End]=ReadTrainData(fd_test_source,fd_test_target,parameter,32);   %transform data to batches
        if End~=1 || (End==1&& length(batch.Word)~=0)   
            [source_h,lstm,h_t,c]=Forward(batch,parameter,0,0);
            clear lstm;
            %LSTM Forward
            [batch_cost,grad]=softmax(h_t,batch,parameter);      

            clear h_t;
            clear grad;
            cost=cost+batch_cost;
            num_word=num_word+batch.N_word;
        end
        if End==1
            fclose(fd_test_source);
            fclose(fd_test_target);
            break;
        end
    end
    disp(1/exp(-cost/num_word))
end



function[parameter]=ReadParameter(parameter)
    filename=parameter.read_file
    for ll=1:parameter.layer_num
        W_file=strcat(filename,'_W_S',num2str(ll));
        parameter.W_S{ll}=gpuArray(load(W_file));
        W_file=strcat(filename,'_W_T',num2str(ll));
        parameter.W_T{ll}=gpuArray(load(W_file));
    end
    v_file=strcat(filename,'_v');
    parameter.vect=gpuArray(load(v_file));
    W_file=strcat(filename,'_soft_W');
    parameter.soft_W=gpuArray(load(W_file));
    
    att_W_file=strcat(filename,'_attention_W');
    dlmwrite(att_W_file,parameter.attention_W);
    
    att_U_file=strcat(filename,'_attention_U');
    dlmwrite(att_U_file,parameter.attention_U);
    
end

function SaveParameter(parameter,iter,f1)
    if iter~=-1
        all=strcat(f1,int2str(iter),'.mat');
    else
        all=f1;
    end
    save('-v7.3',all,'parameter');
end

function[parameter]=update_parameter(parameter,grad)    
    %update parameters
    norm=computeGradNorm(grad,parameter);       
    %compute normalization

    if norm>parameter.maxGradNorm
        lr=parameter.alpha*parameter.maxGradNorm/norm;  
        %normalizing
    else lr=parameter.alpha;
    end
    for ll=1:parameter.layer_num
        parameter.W_T{ll}=parameter.W_T{ll}-lr*grad.W_T{ll};
        parameter.W_S{ll}=parameter.W_S{ll}-lr*grad.W_S{ll};
    end
    parameter.soft_W=parameter.soft_W-lr*grad.soft_W;
    parameter.vect(:,grad.indices)=parameter.vect(:,grad.indices)-lr*grad.W_emb;
    parameter.Atten_W=parameter.Atten_W-lr*grad.Atten_W;
end


function[parameter]=Initial(parameter)
    %random initialization
    m=parameter.Initial;
    for i=1:parameter.layer_num
        if i==1
            parameter.W_S{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,2*parameter.dimension]);
            parameter.W_T{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,3*parameter.dimension]);
        else
            parameter.W_S{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,2*parameter.hidden]);
            parameter.W_T{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,2*parameter.hidden]);
        end
    end
    parameter.Atten_W=randomMatrix(parameter.Initial,[parameter.dimension,2*parameter.dimension]);
    parameter.vect=randomMatrix(parameter.Initial,[parameter.dimension,parameter.Vocab]);
    parameter.soft_W=randomMatrix(parameter.Initial,[parameter.TargetVocab,parameter.hidden]);
end

function[current_batch,End]=ReadTrainData(fd_s,fd_t,parameter,batch_num)
    if batch_num==0
        batch_N=parameter.batch_size;
    else
        batch_N=batch_num;
    end
    tline_s = fgets(fd_s);
    tline_t = fgets(fd_t);
    i=0;
    Source={};Target={};
    End=0;
    while ischar(tline_s)
        i=i+1;
        text_s=deblank(tline_s);
        text_t=deblank(tline_t);
        if parameter.reverse==1
            Source{i}=wrev(str2num(text_s))+parameter.TargetVocab;  
        else 
            Source{i}=wrevstr2num(text_s)+parameter.TargetVocab;
        end
        Target{i}=[str2num(text_t),parameter.stop];     
        %add document_end_token
        if i==batch_N
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
    current_batch.Mask=ones(N,total_length);
    % Mask: labeling positions where no words exisit. The purpose is to work on sentences in bulk making program faster
    for j=1:N
        source_length=length(Source{j});
        target_length=length(Target{j});
        current_batch.Word(j,current_batch.MaxLenSource-source_length+1:current_batch.MaxLenSource)=Source{j};      
        %words within sentences 
        current_batch.Word(j,current_batch.MaxLenSource+1:current_batch.MaxLenSource+target_length)=Target{j};
        current_batch.Mask(j,1:current_batch.MaxLenSource-source_length)=0;       
        current_batch.Mask(j,current_batch.MaxLenSource+target_length+1:end)=0;   
        % label positions without tokens 0
        current_batch.N_word=current_batch.N_word+target_length;
    end
%     current_batch.Word
    for j=1:total_length
        current_batch.Delete{j}=find(current_batch.Mask(:,j)==0);
        current_batch.Left{j}=find(current_batch.Mask(:,j)==1);
    end
end

function[norm]=computeGradNorm(grad,parameter)  %compute gradient norm
    norm=0;
    for ii=1:parameter.layer_num
        norm=norm+double(sum(grad.W_S{ii}(:).^2));
        norm=norm+double(sum(grad.W_T{ii}(:).^2));
    end
    norm=norm+double(sum(grad.soft_W(:).^2));
    norm=norm+double(sum(grad.W_emb(:).^2));
    norm=sqrt(norm);
end
