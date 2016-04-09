function[]=generate_score(isFrench,isDev)
gpuDevice(1);
load '../training/en_given_gr0.2/12.mat';

parameter.train_source_file='rerank/rearank_target_dev';
parameter.train_target_file='rerank/rearank_source_dev';
parameter.write_filename='score/s_given_t';

Get_Score(parameter);

end



