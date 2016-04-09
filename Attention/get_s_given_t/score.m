function[]=score(h_t,batch,parameter,filename)
    Word=batch.Word(:,batch.MaxLenSource+1:end);
    Mask=batch.Mask(:,batch.MaxLenSource+1:end);
    N_examples=size(Word,1)*size(Word,2);
    predict_Words=reshape(Word,1,N_examples);
    mask=reshape(Mask,1,N_examples);
    h_t=[h_t{:}];
    scores=batchSoftmax(h_t,mask,predict_Words,parameter);

    scores=reshape(scores,size(Word,1),size(Word,2)).*Mask;
   %-log2(exp(-scores))
    scores=sum(scores,2)./sum(Mask,2);
    %-log2(exp(-scores))G
    dlmwrite(filename,scores,'-append');
    clear predict_Words; clear mask;
end

function[scores]=batchSoftmax(h_t,mask,predict_Words,parameter)
%softmax matrix
    maskedIds=find(mask==0);
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
     scores=log(scores);

    scoreIndices=sub2ind(size(scores),predict_Words,1:length(predict_Words));
    scores=scores(scoreIndices);
    scores(maskedIds)=0;
    clear norms;
end
