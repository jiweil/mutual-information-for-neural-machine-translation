function[parameter]=Parameter(parameter)
addpath('../../misc');

parameter.isGPU = 1;

parameter.dimension=1000;
parameter.alpha=1;    %learning rate
parameter.fix_alpha=parameter.alpha;
parameter.layer_num=4;  %number of layer
parameter.hidden=1000;
parameter.isReverse=1;
parameter.lstm_out_tanh=1;
parameter.Initial=0.1;
parameter.dropout=0.2;  %drop-out rate
parameter.isTraining=0;
parameter.CheckGrad=0;  %whether check gradient or not.
parameter.PreTrainEmb=0;    %whether using pre-trained embeddings
parameter.update_embedding=1;   %whether update word embeddings
parameter.batch_size=128;    %mini-batch size
%whether source and target is of the same language. For author-encoder task, it is.
parameter.maxGradNorm=5;    %gradient clipping
parameter.clip=0;
parameter.Source_Target_Same_Language=0;

parameter.lr=5;

parameter.SourceVocab=50000;
parameter.TargetVocab=50001;
parameter.Vocab=parameter.SourceVocab+parameter.TargetVocab;
parameter.stop=50001;

parameter.nonlinear_gate_f = @sigmoid;
parameter.nonlinear_gate_f_prime = @sigmoidPrime;
parameter.nonlinear_f = @tanh;
parameter.nonlinear_f_prime = @tanhPrime;


parameter.french=0;
parameter.TargetEn=0;

if parameter.french==1 && parameter.TargetEn==0
    parameter.save_folder='fr_given_en0.2/';
    parameter.train_source_file='../data_fr/en'
    parameter.train_target_file='../data_fr/fr'
    parameter.test_source_file='../../data_fr/test_en';
    parameter.test_target_file='../../data_fr/test_fr';
    N_line=11990624;
    parameter.SourceVocab=200000;
    parameter.TargetVocab=80002;
    parameter.Vocab=parameter.SourceVocab+parameter.TargetVocab;
    parameter.left_stop=80001;
    parameter.right_stop=80002;
    parameter.start_half_iter=5;
end
if parameter.french==0 &&parameter.TargetEn==0
    parameter.save_folder='gr_given_en0.2/';
    train_source_file='../data_gr/en'
    train_target_file='../data_gr/gr'
    parameter.test_source_file='../../data_gr/valid_en';
    parameter.test_target_file='../../data_gr/valid_gr';
    N_line=4468840;
    parameter.SourceVocab=50000;
    parameter.TargetVocab=50001;

    parameter.Vocab=parameter.SourceVocab+parameter.TargetVocab;
    parameter.stop=50001;
    parameter.start_half_iter=8;
end

if parameter.french==0 &&parameter.TargetEn==1
    parameter.save_folder='en_given_gr0.2/';
    train_source_file='../data_gr/gr'
    train_target_file='../data_gr/en'
    parameter.test_source_file='../data_gr/valid_gr';
    parameter.test_target_file='../data_gr/valid_en';
    N_line=4468840;
    parameter.SourceVocab=50000;
    parameter.TargetVocab=50001;
    parameter.Vocab=parameter.SourceVocab+parameter.TargetVocab;
    parameter.stop=50001;
    parameter.start_half_iter=8;
end

end
