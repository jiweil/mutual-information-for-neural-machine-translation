function[grad]=Backward(source_lstm,h_t,batch,grad,parameter,lstms,all_c_t)%backward
    dh = cell(parameter.layer_num, 1);
    dc = cell(parameter.layer_num, 1);
    Word=batch.Word;
    N=size(Word,1);
    T=batch.MaxLen;
    MaxLenSource=batch.MaxLenSource;
    MaxLenTarget=batch.MaxLenTarget;
    for ll=1:parameter.layer_num
        grad.W_T{ll}=zeroMatrix(size(parameter.W_T{ll}));
        grad.W_S{ll}=zeroMatrix(size(parameter.W_S{ll}));
    end
    grad.Atten_W=zeroMatrix(size(parameter.Atten_W));

    zeroState=zeroMatrix([parameter.hidden,size(batch.Word,1)]);
    for ll=parameter.layer_num:-1:1
        dh{ll} = zeroState;
        dc{ll} = zeroState;
    end
    wordCount = 0;
    numInputWords=size(Word,1)*size(Word,2);
    allEmbGrads=zeroMatrix([parameter.dimension,numInputWords]);
    
    d_source_v=gpuArray();
    for t=T-1:-1:1
        unmaskedIds=batch.Left{t};
        for ll=parameter.layer_num:-1:1
            if ll==parameter.layer_num &&t>=MaxLenSource
                atten=lstms{ll,t}.atten;
                if t==T-1
                    d_h_inside=arrayfun(@tanhPrimeDouble,h_t{t-batch.MaxLenSource+1},grad.ht{t});
                else d_h_inside=arrayfun(@tanhPrimeDouble,h_t{t-batch.MaxLenSource+1},grad.ht{t}+d_soft_max);
                end
                grad.Atten_W=grad.Atten_W+d_h_inside*[atten.h;atten.vector]';
                d_v=parameter.Atten_W'*d_h_inside;
                clear d_h_inside;
                dh{ll}=dh{ll}+d_v(1:parameter.dimension,:);

                d_atten=d_v(1+parameter.dimension:2*parameter.dimension,:);
                [d_current_source_v,d_current_h]=BackwardAtten(d_atten,batch,atten,source_lstm);
                clear d_atten;
                if length(d_source_v)==0 d_source_v=d_current_source_v;
                else d_source_v=d_source_v+d_current_source_v;
                end
                dh{ll}=dh{ll}+d_current_h;
                clear d_current_source_v;
                clear d_cuurent_h;
                clear d_v;
            end
            if t==MaxLenSource&&ll==parameter.layer_num
                d_source_v=permute(d_source_v,[1,3,2]);
            end
            if ll==parameter.layer_num&&t<=MaxLenSource
                dh{ll}=dh{ll}+d_source_v(:,:,t);
            end
            if t==1 c_t_1 = [];
            else c_t_1 = all_c_t{ll, t-1};
            end
            c_t = all_c_t{ll, t};
            lstm = lstms{ll, t};
            [lstm_grad]=lstmUnitGrad(lstm, c_t, c_t_1, dc{ll}, dh{ll},ll, t,zeroState,parameter,MaxLenSource);%backward propagation
            if (t>MaxLenSource)grad.W_T{ll}=grad.W_T{ll}+lstm_grad.W;
            else grad.W_S{ll}=grad.W_S{ll}+lstm_grad.W;
            end
            dc{ll} = lstm_grad.dc;
            dh{ll} = lstm_grad.input(end-parameter.hidden+1:end,:);
            if ll==1&&t>batch.MaxLenSource
                d_soft_max=lstm_grad.input(parameter.hidden+1:2*parameter.hidden,:);
            end
            if ll==1%deal with word emebddings
                embIndices=batch.Word(unmaskedIds,t)';
                embGrad = lstm_grad.input(1:parameter.dimension,unmaskedIds);
                numWords = length(embIndices);
                allEmbIndices(wordCount+1:wordCount+numWords) = embIndices;
                allEmbGrads(:, wordCount+1:wordCount+numWords) = embGrad;
                wordCount = wordCount + numWords;
            else
                dh{ll-1}(:,unmaskedIds)=dh{ll-1}(:,unmaskedIds)+lstm_grad.input(1:parameter.hidden,unmaskedIds);
            end
            clear lstms{ll,t};
        end
    end
    allEmbGrads(:, wordCount+1:end) = [];
    allEmbIndices(wordCount+1:end) = [];
    [grad.W_emb, grad.indices] = aggregateMatrix(allEmbGrads, allEmbIndices);
    for ll=1:parameter.layer_num
        grad.W_T{ll}=grad.W_T{ll}/N;%divided by number of sentences
        grad.W_S{ll}=grad.W_S{ll}/N;
    end
    grad.Atten_W=grad.Atten_W/N;
    grad.W_emb=grad.W_emb/N;
    grad.soft_W=grad.soft_W/N;
    clear allEmbGrads;
    clear allEmbIndices;
    clear allEmbGrads;
end

function[lstm_grad]=lstmUnitGrad(lstm, c_t, c_t_1, dc, dh, ll, t, zero_state,parameter,MaxLenSource)
    dc =arrayfun(@plusTanhPrimeTriple,dc,lstm.f_c_t,lstm.o_gate, dh);
    %dc = arrayfun(@plusMult, dc, lstm.o_gate, dh);
    do = arrayfun(@sigmoidPrimeTriple, lstm.o_gate, lstm.f_c_t, dh);
    di = arrayfun(@sigmoidPrimeTriple, lstm.i_gate, lstm.a_signal, dc);

    if t>1 
        df = arrayfun(@sigmoidPrimeTriple, lstm.f_gate, c_t_1, dc);
    else 
        df = zero_state;
    end
    lstm_grad.dc = lstm.f_gate.*dc;
    dl = arrayfun(@tanhPrimeTriple, lstm.a_signal, lstm.i_gate, dc);
    d_ifoa = [di; df; do; dl];
    lstm_grad.W = d_ifoa*lstm.input'; %dw
    if t>MaxLenSource
        lstm_grad.input = parameter.W_T{ll}'*d_ifoa;% dx dh
    else
        lstm_grad.input = parameter.W_S{ll}'*d_ifoa;
    end
    if parameter.dropout~=0
        %lstm_grad.input(1:parameter.hidden, :) = lstm_grad.input(1:parameter.hidden, :).*lstm.drop_left;
         lstm_grad.input=lstm_grad.input.*lstm.drop_left;
    end
    if parameter.clip==1%if values of gradients are too large, clip them
        lstm_grad.input=arrayfun(@clipBackward, lstm_grad.input);
        lstm_grad.dc = arrayfun(@clipBackward, lstm_grad.dc);
    end
    clear dc; clear do; clear di; clear df; clear d_ifoa;
end


function [value] = plusTanhPrimeTriple(t, x, y, z)
    value = t + (1-x*x)*y*z;
end
function [value] = tanhPrimeTriple(x, y, z)
    value = (1-x*x)*y*z;
end
function [value] = plusMult(x, y, z)
    value = x + y*z;
end
function [value] = sigmoidPrimeTriple(x, y, z)
    value = x*(1-x)*y*z;
end
function [value] = tanhPrimeDouble(x, y)
    value = (1-x*x)*y;
end

function [clippedValue] = clipBackward(x)
    if x>1000 clippedValue = single(1000);
    elseif x<-1000 clippedValue = single(-1000);
    else clippedValue =single(x);
    end
end
