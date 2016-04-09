function[source_lstm,lstms,h_t,all_c_t]=Forward(batch,parameter,isTraining,isDecoding)%Forward
    N=size(batch.Word,1);
    T=size(batch.Word,2);
    zeroState=zeroMatrix([parameter.hidden,N]);
    all_c_t=cell(parameter.layer_num,T);
    lstms = cell(parameter.layer_num,T);
    source_lstm=gpuArray();
    
    for ll=1:parameter.layer_num
        for tt=1:T
            all_c_t{ll,tt}=zeroMatrix([parameter.hidden,N]);
        end
    end
    source_lstm=zeroMatrix([parameter.dimension,N,batch.MaxLenSource]);
    for t=1:batch.MaxLenSource
        for ll=1:parameter.layer_num
            W=parameter.W_S{ll};
            if t==1
                h_t_1=zeroState;
                c_t_1 =zeroState;
            else
                h_t_1 = h{ll};
                c_t_1 = all_c_t{ll, t-1};
            end
            if ll==1
                x_t=parameter.vect(:,batch.Word(:,t));
            else
                x_t=h{ll-1};
            end
            x_t(:,batch.Delete{t})=0;
            h_t_1(:,batch.Delete{t})=0;
            c_t_1(:,batch.Delete{t})=0;
            [lstms{ll, t},h{ll},all_c_t{ll, t}]=lstmUnit(W,parameter,x_t,h_t_1,c_t_1,ll,t,isTraining);%LSTM unit calculation
            if ll==parameter.layer_num
                source_lstm(:,:,t)=h{ll};
            end
            if ll==parameter.layer_num&&t==batch.MaxLenSource
                source_lstm=permute(source_lstm,[1,3,2]);

                atten=WorkAtten(source_lstm,h{ll},batch);
                h_t{1}=parameter.nonlinear_f(parameter.Atten_W*[h{ll};atten.vector]);
                atten.h=h{ll};
                lstms{ll, t}.atten=atten;
            end
        end
    end

    for t=1+batch.MaxLenSource:batch.MaxLen-1
        for ll=1:parameter.layer_num
            W=parameter.W_T{ll};
            h_t_1 = h{ll};
            c_t_1 = all_c_t{ll, t-1};
            if ll==1
                x_t=parameter.vect(:,batch.Word(:,t));
                x_t=[x_t;h_t{t-batch.MaxLenSource}];
            else
                x_t=h{ll-1};
            end
            x_t(:,batch.Delete{t})=0;
            h_t_1(:,batch.Delete{t})=0;
            c_t_1(:,batch.Delete{t})=0;
            [lstms{ll, t},h{ll},all_c_t{ll, t}]=lstmUnit(W,parameter,x_t,h_t_1,c_t_1,ll,t,isTraining);%LSTM unit calculation
            if ll==parameter.layer_num
                atten=WorkAtten(source_lstm,h{ll},batch);
                h_t{t-batch.MaxLenSource+1}=parameter.nonlinear_f(parameter.Atten_W*[h{ll};atten.vector]);
                atten.h=h{ll};
                lstms{ll, t}.atten=atten;

                clear atten;
            end
        end
    end
    if isDecoding==1
        h_t=h;
        all_c_t=all_c_t(:,batch.MaxLenSource);
    end
end
