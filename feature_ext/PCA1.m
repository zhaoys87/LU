function [dataTrans, eigVal, eigVec] = PCA1( data, lowDim)
% PCA transformation
% input parameter:
%   data :  input data with M * N * bands
%   lowDim :  the low dimension after the reduction
% output parameter: output pca transformation features
%   dataTrans : output the transformed data with M * N * lowDim ;
%   eigVal : output the eigenvalues of the covariance ;
%   eigVec: output the corresponding eigenvectors;
% Writed by Bei Zhao on Apr. 10th, 2014.

[rows, columns, org_dim] = size(data);
if(~exist('lowDim','var') || lowDim > org_dim)
    lowDim = org_dim;
end

%% cal the covariance matrix
data_num = rows * columns;
feaArr = double( reshape(data, [data_num, org_dim]) );
expX = sum(feaArr,1) / data_num;
cov_X = feaArr' * feaArr;
cov_X = cov_X /  data_num - expX' * expX;

%% eigenvector computation
[eigVec, eigVal] = eig(cov_X);
eigVal = diag(eigVal) ;
[eigVal,eigIDX] = sort(eigVal,'descend') ;
eigVec = eigVec(:,eigIDX);

if(lowDim < 1 && lowDim > 0)
    cum = cumsum(eigVal);
    cum = cum / cum(end);
    ids = cum > lowDim;
    [~, ia] = unique(ids);
    lowDim = ia(2);
end

%% projection to principle component
dataTrans  =  bsxfun(@minus, feaArr, expX) * eigVec(:,1:lowDim);
dataTrans = reshape( dataTrans, [rows ,columns, lowDim]);

end


