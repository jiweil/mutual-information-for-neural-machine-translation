function[]=LSTM(targetIsEn)
addpath('../../misc');
n= gpuDeviceCount;

parameter.TargetEn=targetIsEn;

parameter.isGPU = 1;
gpu_index=1
gpuDevice(gpu_index);

parameter.dimension=1000;
parameter.alpha=1;    %learning rate
parameter.fix_alpha=parameter.alpha;
parameter.layer_num=4;  %number of layer
parameter.hidden=1000;
parameter.isReverse=1;
parameter.lstm_out_tanh=1;
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
parameter.Source_Target_Same_Language=0;

parameter.lr=5;


if parameter.CheckGrad==1&parameter.dropout~=0  %use identical dropout-vector for gradient checking
    parameter.drop_left=randSimpleMatrix([2*parameter.hidden,1])<1-parameter.dropout;
end

parameter.nonlinear_gate_f = @sigmoid;
parameter.nonlinear_gate_f_prime = @sigmoidPrime;
parameter.nonlinear_f = @tanh;
parameter.nonlinear_f_prime = @tanhPrime;

if parameter.TargetEn==0
    parameter.save_folder='gr_given_en0.2/';
    train_source_file='../../data_gr/train_en'
    train_target_file='../../data_gr/train_gr'
    test_source_file='../../data_gr/dev_en'
    test_target_file='../../data_gr/dev_gr'
    N_line=4468840;
    parameter.SourceVocab=50000;
    parameter.TargetVocab=50001;
    parameter.Vocab=parameter.SourceVocab+parameter.TargetVocab;
    parameter.stop=parameter.TargetVocab;
    parameter.start_half_iter=8;
end

if parameter.TargetEn==1
    parameter.save_folder='en_given_gr0.2/';
    train_source_file='../../data_gr/train_gr'
    train_target_file='../../data_gr/train_en'
    test_source_file='../../data_gr/dev_gr'
    test_target_file='../../data_gr/dev_en'
    N_line=4468840;
    parameter.SourceVocab=50000;
    parameter.TargetVocab=50001;
    parameter.Vocab=parameter.SourceVocab+parameter.TargetVocab;
    parameter.stop=parameter.TargetVocab;
    parameter.start_half_iter=8;
end

% above is files for gradient checking

iter=0;
parameter.read=0;
parameter=Initial(parameter)

disp('begin')
num_batch=N_line/parameter.batch_size

parameter
while 1
    iter=iter+1
    End=0;
    fd_train_source=fopen(train_source_file);   %read source
    fd_train_target=fopen(train_target_file);   %read target
    sum_cost=0;
    sum_num=0;
    batch_n=0;
    if iter>parameter.start_half_iter
        parameter.alpha=parameter.alpha*0.5;
        disp('learning rate');
        parameter.alpha
    end
    tic
    while 1 
        batch_n=batch_n+1;
        [batch,End]=ReadTrainData(fd_train_source,fd_train_target,parameter,1);   %transform data to batches
        if size(batch.Word,2)>100
            continue;
        end
        if End~=1 || (End==1&& length(batch.Word)~=0)   
            %not the end of document
            [lstm,h_t,c]=Forward(batch,parameter,1,0);    
            %LSTM Forward
            [batch_cost,grad]=softmax(h_t,batch,parameter,1);      

            %disp(1/exp(-batch_cost/batch.N_word))
            clear h_t;
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
                grad=Backward(batch,grad,parameter,lstm,c);     
                if 1==parameter.CheckGrad
                    check(grad,batch,parameter)
                end
                %backward propagation
                [parameter]=update_parameter(parameter,grad);   
                %update parameter
                clear lstm;
                clear c;
                clear grad;
            end
        end
        if End==1
            fclose(fd_train_source);
            fclose(fd_train_target);
            break;
        end
        if 1==1
        if mod(batch_n,20000)==0
            batch_n
            testing(test_source_file,test_target_file,parameter);
        end
        end
    end
    clear batch;
    iter
    SaveParameter(parameter,iter,parameter.save_folder);  
    testing(test_source_file,test_target_file,parameter);
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
        [batch,End]=ReadTrainData(fd_test_source,fd_test_target,parameter,0);   %transform data to batches
        if End~=1 || (End==1&& length(batch.Word)~=0)   
            [lstm,h_t,c]=Forward(batch,parameter,0,0);    
            clear lstm;
            %LSTM Forward
            [batch_cost,grad]=softmax(h_t,batch,parameter,0);      

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
    disp('read_folder')
    parameter.ReadFile
    filename=parameter.ReadFile;
    for ll=1:parameter.layer_num
        W_file=strcat(filename,'_W_S',num2str(ll));
        parameter.W_S{ll}=gpuArray(load(W_file));
        W_file=strcat(filename,'_W_T',num2str(ll));
        parameter.W_T{ll}=gpuArray(load(W_file));
    end
    v_file=strcat(filename,'_tgt_v');
    tgt_v=gpuArray(load(v_file));
    v_file=strcat(filename,'_src_v');
    src_v=gpuArray(load(v_file));
    parameter.vect=[tgt_v,src_v];
    clear src_v;
    clear tgt_v;
    W_file=strcat(filename,'_soft_W');
    parameter.soft_W=gpuArray(load(W_file));
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
end

function[parameter]=Initial(parameter)
    %random initialization
    m=parameter.Initial;
    for i=1:parameter.layer_num
        parameter.W_S{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,2*parameter.hidden]);
        parameter.W_T{i}=randomMatrix(parameter.Initial,[4*parameter.hidden,2*parameter.hidden]);
    end
    parameter.vect=randomMatrix(parameter.Initial,[parameter.dimension,parameter.Vocab]);
    parameter.soft_W=randomMatrix(parameter.Initial,[parameter.TargetVocab,parameter.hidden]);
end

function[current_batch,End]=ReadTrainData(fd_s,fd_t,parameter,isTraining)
    if isTraining==1
        batch_size=parameter.batch_size;
    else 
        batch_size=32;
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
        if parameter.isReverse==1
            Source{i}=wrev(str2num(text_s))+parameter.TargetVocab;  
        else Source{i}=str2num(text_s)+parameter.TargetVocab;
        end
            %reverse inputs
        Target{i}=[str2num(text_t),parameter.stop];     
        %add document_end_token
        if i==batch_size
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

function check(grad,batch,parameter)
    %check_soft_W(grad.soft_W(1,1),1,1,batch,parameter);
    check_target_W(grad.W_T{1}(parameter.hidden+1,1),1,parameter.hidden+1,1,batch,parameter);
    check_source_W(grad.W_S{1}(parameter.hidden+1,1),1,parameter.hidden+1,1,batch,parameter);
    check_target_W(grad.W_T{1}(1,1),1,1,1,batch,parameter);
    check_source_W(grad.W_S{1}(1,1),1,1,1,batch,parameter);
end

%gradient check
function check_soft_W(value1,i,j,batch,parameter)
    e=0.001;
    parameter.soft_W(i,j)=parameter.soft_W(i,j)+e;
    [lstms,h,c]=Forward(batch,parameter,1,0);
    [cost1,grad]=softmax(h(parameter.layer_num,:),batch,parameter,1);
    parameter.soft_W(i,j)=parameter.soft_W(i,j)-2*e;
    [lstms,h,c]=Forward(batch,parameter,1,0);
    [cost2,grad]=softmax(h(parameter.layer_num,:),batch,parameter,1);
    value2=(cost1-cost2)/(2*e);
    value1
    value2
    value1-value2
end


function check_target_W(value1,ll,i,j,batch,parameter)
    e=0.001;
    parameter.W_T{ll}(i,j)=parameter.W_T{ll}(i,j)+e;
    [lstm,h_t,c]=Forward(batch,parameter,1,0);    
    [cost1,grad]=softmax(h_t,batch,parameter,1);      
    parameter.W_T{ll}(i,j)=parameter.W_T{ll}(i,j)-2*e;
    [lstm,h_t,c]=Forward(batch,parameter,1,0);    
    [cost2,grad]=softmax(h_t,batch,parameter,1);      
    parameter.W_T{ll}(i,j)=parameter.W_T{ll}(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    value1
    value2
    value1-value2
end

function check_source_W(value1,ll,i,j,batch,parameter)
    e=0.001;
    parameter.W_S{ll}(i,j)=parameter.W_S{ll}(i,j)+e;
    [lstm,h_t,c]=Forward(batch,parameter,1,0);    
    [cost1,grad]=softmax(h_t,batch,parameter,1);      
    parameter.W_S{ll}(i,j)=parameter.W_S{ll}(i,j)-2*e;
    [lstm,h_t,c]=Forward(batch,parameter,1,0);    
    [cost2,grad]=softmax(h_t,batch,parameter,1);      
    parameter.W_S{ll}(i,j)=parameter.W_S{ll}(i,j)+e;
    cost1
    cost2
    value2=(cost1-cost2)/(2*e);
    value1
    value2
    value1-value2
end


function check_vect(value1,i,j,batch,parameter)
    e=0.001;
    parameter.vect(i,j)=parameter.vect(i,j)+e;
    [lstms,h,c]=Forward(batch,parameter,1);
    [cost1,grad]=softmax(h(parameter.layer_num,:),batch,parameter);
    parameter.vect(i,j)=parameter.vect(i,j)-2*e;
    [lstms,h,c]=Forward(batch,parameter,1);
    [cost2,grad]=softmax(h(parameter.layer_num,:),batch,parameter);
    parameter.vect(i,j)=parameter.vect(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    value1
    value2
    value1-value2
end

function[norm]=computeGradNorm(grad,parameter)  %compute gradient norm
    norm=0;
    for ii=1:parameter.layer_num
        norm=norm+double(sum(grad.W_S{ii}(:).^2));
        norm=norm+double(sum(grad.W_T{ii}(:).^2));
    end
    norm=norm+double(sum(grad.soft_W(:).^2));
    norm=sqrt(norm);
end
