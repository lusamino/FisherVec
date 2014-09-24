% The dimensions of the inputs are:
% inputW = K x 1 / inputX = T x D / inputMean = K x D / inputCov = K x D

function [fisher_vector,fisher_vector_norm]=fisher_vector_calc(inputW,inputX,inputMean,inputCov)

D=size(inputMean,2);
K=size(inputMean,1);
T=size(inputX,1);

% Calculation of the gamma from the data image

gamma_res=gamma_calc(inputW,inputX,inputMean,inputCov);
% gamma_res is T x K

% Computation of values of S
S0=sum(gamma_res,1); % 1 x K

tensor_gamma=[];
for i=1:D
    tensor_gamma=cat(3,tensor_gamma,gamma_res);
end;
aux_X=reshape(inputX,T,1,D);
tensor_X=[];
for i=1:K
    tensor_X=cat(2,tensor_X,aux_X);
end;

S1=sum(tensor_gamma.*tensor_X,1);
S1=reshape(S1,K,D); % K x D

S2=sum(tensor_gamma.*(tensor_X.^2),1);
S2=reshape(S2,K,D); % K x D

% Compute the gradients from the values of S

gradAlpha=(S0'-T*inputW)./sqrt(inputW); % K x 1

gradMean=(S1-inputMean.*kron(S0',ones(1,D)))./(kron(sqrt(inputW),ones(1,D)).*sqrt(inputCov)); % K x D

gradCov=(S2-2*inputMean.*S1+(inputMean.^2-inputCov).*kron(S0',ones(1,D)))./...
    (sqrt(2)*kron(sqrt(inputW),ones(1,D)).*inputCov); % K x D

[gradAlpha_alt,gradMean_alt,gradCov_alt]=other_compGrads(gamma_res,inputW,inputX,inputMean,inputCov);
% sum(sum(abs(gradAlpha-gradAlpha_alt)))
% sum(sum(abs(gradMean-gradMean_alt)))
% sum(sum(abs(gradCov-gradCov_alt)))

% Concatenate the Fisher vector

fisher_vector=gradAlpha;
fisher_vector=cat(1,fisher_vector,reshape(gradMean',K*D,1));
fisher_vector=cat(1,fisher_vector,reshape(gradCov',K*D,1));

% Apply power Normalization

fisher_vector_norm=sign(fisher_vector).*sqrt(abs(fisher_vector));

% Apply L2 normalization
fisher_vector_norm=fisher_vector_norm./sqrt(fisher_vector_norm'*fisher_vector_norm);
fisher_vector_norm(isnan(fisher_vector_norm))=0;