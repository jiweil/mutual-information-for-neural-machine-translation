function[total_cost,grad]=softmax(h,batch,parameter)
    MaxLenSource=batch.MaxLenSource;
    MaxLenTarget=batch.MaxLenTarget;
    MaxLen=batch.MaxLen;
    Word=batch.Word;
    N=size(Word,1);
    Mask=batch.Mask;

    step_size=ceil(500/size(Word,1));%calculate softmax in bulk
    total_cost=0;
    grad.soft_W=zeroMatrix(size(parameter.soft_W));
    for Begin=1:MaxLenTarget        
        N_examples=size(Word,1);
        predict_Words=reshape(Word(:,MaxLenSource+Begin),1,N_examples);
        mask=reshape(Mask(:,MaxLenSource+Begin),1,N_examples);%remove positions with no words
        h_t=[h{:,Begin}];
        [cost,grad_softmax_h]=batchSoftmax(h_t,mask,predict_Words,parameter);
        total_cost=total_cost+cost;
        grad.soft_W=grad.soft_W+grad_softmax_h.soft_W;
        grad.ht{1,Begin+MaxLenSource-1}=grad_softmax_h.h; 
    end
    if parameter.CheckGrad==1
        total_cost=total_cost/size(Word,1);
    end
    clear predict_Words; clear mask;
    clear grad_softmax_h;
end

function[cost,softmax_grad]=batchSoftmax(h_t,mask,predict_Words,parameter)
%softmax matrix
    unmaskedIds=find(mask==1);
    scores=parameter.soft_W*h_t;
    mx = max(scores,[],1);
    scores=bsxfun(@minus,scores,mx);
    scores=exp(scores);
    norms = sum(scores, 1);
    if length(find(mask==0))==0
        scores=bsxfun(@rdivide, scores, norms);
    else
        scores=bsxfun(@times,scores, mask./norms); 
    end
    scoreIndices = sub2ind(size(scores),predict_Words(unmaskedIds),unmaskedIds);
    cost=sum(-log(scores(scoreIndices)));
    scores(scoreIndices) =scores(scoreIndices) - 1;
    softmax_grad.soft_W=scores*h_t';  %(N_word*examples)*(examples*diemsnion)=N_word*diemsnion;
    softmax_grad.h=(scores'*parameter.soft_W)';%(diemsnion*N_word)*(N_word*examples)=dimension*examples
    clear scores;
    clear norms;
end
