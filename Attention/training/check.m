function check(grad,batch,parameter)
    if 1==1
    disp('check_target_W')
    check_target_W(grad.W_T{1}(1,1),1,1,1,batch,parameter);
    check_target_W(grad.W_T{1}(1+parameter.dimension,1),1,1+parameter.dimension,1,batch,parameter);
    check_target_W(grad.W_T{1}(1+2*parameter.dimension,1),1,1+2*parameter.dimension,1,batch,parameter);
    check_target_W(grad.W_T{2}(1,1),2,1,1,batch,parameter);
    check_target_W(grad.W_T{3}(1,1),3,1,1,batch,parameter);
    check_target_W(grad.W_T{4}(1,1),4,1,1,batch,parameter);
    end
    if 1==1
    disp('check_soft_W');
    check_soft_W(grad.soft_W(1,1),1,1,batch,parameter);
    end
    if 1==1
    disp('check_atten_W');
    check_Atten_W(grad.Atten_W(1,1),1,1,batch,parameter);
    end
    if 1==1
    disp('check_source_W');
    check_source_W(grad.W_S{1}(1,1),1,1,1,batch,parameter);
    check_source_W(grad.W_S{2}(1,1),2,1,1,batch,parameter);
    check_source_W(grad.W_S{3}(1,1),3,1,1,batch,parameter);
    check_source_W(grad.W_S{4}(1,1),4,1,1,batch,parameter);
    end
    if 1==1
    disp('check_v')
    check_vect(grad.W_emb(1,1),1,grad.indices(1,1),batch,parameter);
    check_vect(grad.W_emb(1,2),1,grad.indices(1,2),batch,parameter);
    end
end


%gradient check
function check_Atten_W(value1,i,j,batch,parameter)
    e=0.001;
    parameter.Atten_W(i,j)=parameter.Atten_W(i,j)+e;
    [source_h,lstm,h_t,c]=Forward(batch,parameter,1,0);
    [cost1,grad]=softmax(h_t,batch,parameter);
    parameter.Atten_W(i,j)=parameter.Atten_W(i,j)-2*e;
    [source_h,lstm,h_t,c]=Forward(batch,parameter,1,0);
    [cost2,grad]=softmax(h_t,batch,parameter);
    parameter.Atten_W(i,j)=parameter.Atten_W(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    value1
    value2
    value1-value2
end



function check_soft_W(value1,i,j,batch,parameter)
    e=0.001;
    parameter.soft_W(i,j)=parameter.soft_W(i,j)+e;
    [source_h,lstm,h_t,c]=Forward(batch,parameter,1,0);
    [cost1,grad]=softmax(h_t,batch,parameter);
    parameter.soft_W(i,j)=parameter.soft_W(i,j)-2*e;
    [source_h,lstm,h_t,c]=Forward(batch,parameter,1,0);
    [cost2,grad]=softmax(h_t,batch,parameter);      
    value2=(cost1-cost2)/(2*e);
    value1
    value2
    value1-value2
end


function check_target_W(value1,ll,i,j,batch,parameter)
    e=0.0001;
    parameter.W_T{ll}(i,j)=parameter.W_T{ll}(i,j)+e;
    [source_h,lstm,h_t,c]=Forward(batch,parameter,1,0);
    [cost1,grad]=softmax(h_t,batch,parameter);

    
    parameter.W_T{ll}(i,j)=parameter.W_T{ll}(i,j)-2*e;
    [source_h,lstm,h_t,c]=Forward(batch,parameter,1,0);
    [cost2,grad]=softmax(h_t,batch,parameter);
    parameter.W_T{ll}(i,j)=parameter.W_T{ll}(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    value1
    value2
    value1-value2
end

function check_source_W(value1,ll,i,j,batch,parameter)
    e=0.0001;
    parameter.W_S{ll}(i,j)=parameter.W_S{ll}(i,j)+e;
    [source_h,lstm,h_t,c]=Forward(batch,parameter,1,0);
    [cost1,grad]=softmax(h_t,batch,parameter);
    parameter.W_S{ll}(i,j)=parameter.W_S{ll}(i,j)-2*e;
    [source_h,lstm,h_t,c]=Forward(batch,parameter,1,0);
    [cost2,grad]=softmax(h_t,batch,parameter);
    parameter.W_S{ll}(i,j)=parameter.W_S{ll}(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    value1
    value2
    value1-value2
end


function check_vect(value1,i,j,batch,parameter)
    disp('check_vect')
    e=0.001;
    parameter.vect(i,j)=parameter.vect(i,j)+e;

    [source_h,lstm,h_t,c]=Forward(batch,parameter,1,0);   
    %LSTM Forward
    [cost1,grad]=softmax(h_t,batch,parameter);
    parameter.vect(i,j)=parameter.vect(i,j)-2*e;
    [source_h,lstm,h_t,c]=Forward(batch,parameter,1,0);   
    [cost2,grad]=softmax(h_t,batch,parameter);
    parameter.vect(i,j)=parameter.vect(i,j)+e;
    value2=(cost1-cost2)/(2*e);
    value1
    value2
    value1-value2
end
