% The dimensions are:
% K = number of centroid / T = number of samples / D = dimension of space
% of x
% The dimensions of the inputs are:
% inputW = K x 1 / inputX = T x D / inputMean = K x D / inputCov = K x D
% Here we assume that the covariances matrices are approximated as diagonal
% matrices.
% And the result gamma is a matrix of dimensiones T x K

function gamma_res=gamma_calc(inputW,inputX,inputMean,inputCov)

D=size(inputMean,2);
K=size(inputMean,1);
T=size(inputX,1);

gamma_res=[];
for i_k=1:K
    
    resLike=likelihood_calc(inputX,inputMean,inputCov,D,T,i_k);
    gamma_res=[gamma_res,inputW(i_k)*resLike];
    
end;

gamma_res=gamma_res./(kron(sum(gamma_res,2),ones(1,K)));


function resLike=likelihood_calc(inputX,inputMean,inputCov,D,T,compK)

res=1/((2*pi)^D*sqrt(prod(inputCov(compK,:)))); % res is 1 x 1
resLike=res.*exp(-0.5*sum((inputX-kron(inputMean(compK,:),ones(T,1))).*...
    kron(inputCov(compK,:).^(-1),ones(T,1)).*...
    (inputX-kron(inputMean(compK,:),ones(T,1))),2)); % res is T x 1



