
% Just a function to invoke the rest, load the images and set the
% parameters.

% So just indeed another line to continue playing with github

% And another studid line to continue doing things.

clear all

% addpath('../../SIFT_YantaoNoemie/');
% addpath('../../SIFT_YantaoNoemie/descriptor');
% addpath('../../SIFT_YantaoNoemie/key-location');
% addpath('../../SIFT_YantaoNoemie/match');
% addpath('../../SIFT_YantaoNoemie/orientation');
% addpath('../../SIFT_YantaoNoemie/scale-space');
% addpath('../../SIFT_YantaoNoemie/util');
addpath('../../SIFT_AndreaVedaldi');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters and image to load

imageFile='scene.pgm';
numberCentroids=64; % = K
divide_in_regions=0;
scheme_regions=0;
drawSIFT=1;
numberDimPCA=64;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We load the matrix. If divide_in_regions==1, that matrix is splitted and
% stored in an structure.

I=imreadbw(imageFile);

if divide_in_regions==1
    
    % Here we divide the regions, according to the definitions done before.
    
end;


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feature extraction. In this particular main function, we will simply use
% SIFT to extract the features from every region of the image

%[frames,descriptors,scalespace,difofg]=do_sift(I);
[frames,descriptors,scalespace,difofg]=sift(I);

clear('scalespace','difofg');

% [framesY,descriptorsY,scalespace,difofg]=do_sift(I);
% 
% clear('scalespace','difofg');
%figure
% The columns of the coordinates are change according to those of the
% original SIFT implementation

if drawSIFT==1
    num_vectors=20;
    
    [dum,index_sort]=sort(frames(3,:),'descend');
    frames=frames(:,index_sort);
    colormap('gray');
    imagesc(I);
    h=plotsiftframe(frames(:,1:num_vectors));
end;

% [dum,index_sort]=sort(framesY(3,:),'descend');
% framesY=framesY(:,index_sort);
% figure
% colormap('gray');
% imagesc(I);
% h=plotsiftframe(framesY(:,1:num_vectors));

%load SIFTbyLowe.mat
%showkeys(I,locs(1:num_vectors,:));
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% It seems it is neccesary to reduce the dimensionality through PCA, so
% let's see how many dimensions I will consider in the next steps.
inputX=descriptors';

[coeff,score,eigenvalues]=princomp(inputX);

inputXreduced=score(:,1:numberDimPCA);
fprintf('Total energy of %f \n',sum(eigenvalues(1:numberDimPCA))/sum(eigenvalues));

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We compute the parameters of the GMM. Here we can consider 2 approaches:
% either we fix K and compute the parameters or we automatically look for K
% through other approaches as k-means++
gmfit=gmdistribution.fit(inputXreduced,numberCentroids,'Regularize',1e-4,'CovType','diagonal');
% 'CovType','diagonal' or 'full'
% 'Regularize',1e-5

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now is time for the Fisher vector calculation. We invoke that function
% providing the parameters of our GMM.
% inputW = K x 1 / inputX = T x D / inputMean = K x D / inputCov = K x D
inputW=gmfit.PComponents';
inputMean=gmfit.mu;

auxCov=gmfit.Sigma;

inputCov=zeros(size(inputMean,2),size(inputMean,1));

if size(auxCov,1)>1
    for i=1:size(auxCov,3)
        inputCov(:,i)=diag(auxCov(:,:,i));
    end;
else
    for i=1:size(auxCov,3)
        inputCov(:,i)=(auxCov(1,:,i));
    end;
end;
inputCov=inputCov';

%a=tic;
[fisher_vector,fisher_vector_norm]=fisher_vector_calc(inputW,inputXreduced,inputMean,inputCov);
%toc(a)

% To calculate the distances between centroids:
% a1=kron(inputMean,ones(32,1));a2=kron(ones(32,1),inputMean);dist=a1-a2;dist=dist.^2;dist=sum(dist,2);dist=dist.^0.5;


