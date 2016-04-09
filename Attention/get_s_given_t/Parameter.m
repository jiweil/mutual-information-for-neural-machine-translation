function[parameter]=Parameter(parameter)
addpath('../../misc');
n= gpuDeviceCount;



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
parameter.isTraining=1;
parameter.CheckGrad=0;  %whether check gradient or not.
parameter.PreTrainEmb=0;    %whether using pre-trained embeddings
parameter.update_embedding=1;   %whether update word embeddings
parameter.batch_size=32;    %mini-batch size
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


parameter.french=0;
parameter.TargetEn=1;


parameter.SourceVocab=50000;
parameter.TargetVocab=50001;
parameter.Vocab=parameter.SourceVocab+parameter.TargetVocab;
parameter.stop=parameter.TargetVocab;
end


