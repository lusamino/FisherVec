
% Other way of computing the gradients, directly from formular (16), (17)
% and (18)
% inputW = K x 1 / inputX = T x D / inputMean = K x D / inputCov = K x D
% gamma_res = T x K
% gradAlpha_alt = K x 1 / gradMean_alt and gradCov_alt = K x D

function [gradAlpha_alt,gradMean_alt,gradCov_alt]=other_compGrads(gamma_res,inputW,inputX,inputMean,inputCov)

D=size(inputMean,2);
K=size(inputMean,1);
T=size(inputX,1);

gradMean_alt=zeros(K,D);
gradCov_alt=zeros(K,D);

gradAlpha_alt=gamma_res-kron(inputW',ones(T,1));
gradAlpha_alt=sum(gradAlpha_alt,1);
gradAlpha_alt=gradAlpha_alt./sqrt(inputW');
gradAlpha_alt=gradAlpha_alt';

for i=1:K
    
    aux=inputX-kron(inputMean(i,:),ones(T,1));
    aux=aux./kron(sqrt(inputCov(i,:)),ones(T,1));
    aux=kron(gamma_res(:,i),ones(1,D)).*aux;
    aux=sum(aux,1)/sqrt(inputW(i));
    gradMean_alt(i,:)=aux;
    
    aux=(inputX-kron(inputMean(i,:),ones(T,1))).^2;
    aux=aux./kron(inputCov(i,:),ones(T,1));
    aux=(1/sqrt(2))*(aux-1);
    aux=kron(gamma_res(:,i),ones(1,D)).*aux;
    aux=sum(aux,1)/sqrt(inputW(i));
    gradCov_alt(i,:)=aux;
    
end;