function[d_source_v,dh]=BackwardAtten(d_atten,batch,Atten,source_lstm)
    %source_lstm  1000*16*128
    %d_atten: 1000*128
    d_source_v=pagefun(@mtimes,reshape(d_atten,size(d_atten,1),1,size(d_atten,2)),reshape(Atten.scores,1,size(Atten.scores,1),size(Atten.scores,2)));
    % (1000*1*128) *(1*16*128)=(1000*16*128)

    d_A= pagefun(@mtimes,reshape(d_atten,1,size(d_atten,1),size(d_atten,2)),source_lstm);
    % (1*1000*128) *(1000*16*128)=(1*16*128)
    d_A=reshape(d_A,size(d_A,2),size(d_A,3));
    %16*128
    dp=gpuArray();
    N=size(batch.Word,1);
    for i=1:N
        dp=[dp;d_A(:,i)'*(diag(Atten.scores(:,i))-Atten.scores(:,i)*Atten.scores(:,i)')];
    end
    dp=dp';
    dp=reshape(dp,size(dp,1),1,size(dp,2));
    d_source_v=d_source_v+pagefun(@mtimes,reshape(Atten.h,size(Atten.h,1),1,size(Atten.h,2)),permute(dp,[2,1,3]));

    dh=pagefun(@mtimes,source_lstm,dp);
    dh=reshape(dh,size(dh,1),size(dh,3));
end

