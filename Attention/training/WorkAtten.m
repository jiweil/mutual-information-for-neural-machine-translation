function[atten]=WorkAtten(source_lstm,h_t,batch)
    % source_lstm 1000*16*128
    N=size(batch.Word,1);
    reshape_h_t=reshape(h_t,size(h_t,1),1,size(h_t,2));
    %reshape_h_t 1000*1*16
    scores=pagefun(@mtimes,permute(source_lstm,[2,1,3]),reshape_h_t);
    %(16*1000*128)*(1000*1*128)=16*128

    Matrix=reshape(scores,batch.MaxLenSource,N);
    Matrix=exp(Matrix);

    Matrix=Matrix.*batch.Mask(:,1:batch.MaxLenSource)';
    norms=sum(Matrix);
    atten.scores=bsxfun(@rdivide,Matrix, norms);
    vector=pagefun(@mtimes,source_lstm,reshape(atten.scores,size(atten.scores,1),1,size(atten.scores,2)));
    %(1000*16*128)*(16*1*128)=1000*1*128;
    atten.vector=reshape(vector,size(vector,1),size(vector,3));
    clear scores;
    clear reshape_h_t;
    clear norms;
end
