% function Urban_Image_PCA_3C()
%% Urban_Image_PCA_3C is used to perform PCA transformation, select the first 3 pc, and strech them into 0-255;
% input:
%   image_path : the image path;
% output:
%   out_image_path : the path of the output image
%   out_path : the txt file input for Caffe NN.
%% Write by Bei Zhao, zhaoys@cuhk.edu.hk. Nov. 07, 2016.
dataset = 2 ;
bEnlarged = true;
if(bEnlarged)
    larger_size = 256;
end
if(dataset == 2)
    image_path = '../Scene_HSRI/Wuhan-IKONOS/training_small_images';
    out_image_path = '../Scene_HSRI/Wuhan-IKONOS/ImageSamplesPCA';
    img_dim =  4;
else
    image_path = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/HK-WV3/Sample_Images';
    out_image_path = '../Scene_HSRI/HK-WV3/Testing_Samples';
    img_dim =  8;
end
token = '*.hdr';
outsusp = 'bmp';
pca_lowdim = 3;
addpath('feature_ext');

%% Generating bmp files from orginal images by PCA transformation and stretching
DataInfo = SearchFolder2Big( image_path, token );
image_name = DataInfo.img_name;
class_name = DataInfo.class_name;
class_path = DataInfo.class_path;
class_imgnum = DataInfo.class_imgnum;
class_num = numel(class_imgnum);
numimage = sum(class_imgnum);

% get the pca transformed vector for all data;
disp('calculate the covariance matrix...');
cov_X = zeros(img_dim);
expX = zeros(1,img_dim);
count  = 1;
pxlcount = 0;
for ic = 1 : class_num,
    cur_path = class_path{ic};
    for ii = 1 : class_imgnum(ic),
        samp_name = image_name{count};
        [~,name_split] = fileparts(samp_name);
        samp_image_name = fullfile(cur_path, name_split);
        % read image
        [cur_image, dim] = freadenvi(samp_image_name);
        if(bEnlarged)
            cur_image =  imresize( reshape(cur_image, dim), [larger_size, larger_size] );
            cur_image = reshape(cur_image, [larger_size*larger_size, img_dim]);
        end
        % stat the cov and exp;
        expX = expX + sum(cur_image,1); % feaSet.feaArr dim * numSample
        cov_X = cov_X + cur_image' * cur_image;
        pxlcount = pxlcount + size(cur_image, 1);
        count = count  + 1;
    end
    fprintf('\nStat Cov and Exp: %.1f%%\n', ic * 100 / class_num);
end
expX = expX /  pxlcount;
cov_X = cov_X /  pxlcount - expX' * expX;

% SVD decomposition
disp('eigen value descompositon ...');
[eigVec, eigVal] = eig(cov_X);
eigVal = diag(eigVal) ;
[eigVal,eigIDX] = sort(eigVal,'descend') ;
eigVec = eigVec(:,eigIDX);

% projection to principle component
disp('projecting, stretching, and writing images');
count  = 1;
for ic = 1 : class_num,
    cur_path = class_path{ic};
    cur_classname = class_name{ic};
    for ii = 1 : class_imgnum(ic),
        samp_name = image_name{count};
        [~,name_split] = fileparts(samp_name);
        samp_image_name = fullfile(cur_path, name_split);
        cur_outpath = fullfile(out_image_path, cur_classname);
        cur_outpathname = fullfile(cur_outpath, [name_split, '.', outsusp]);
        % read image
        [cur_image, dim] = freadenvi(samp_image_name);
        if(bEnlarged)
            cur_image =  imresize( reshape(cur_image, dim), [larger_size, larger_size] );
            cur_image = reshape(cur_image, [larger_size*larger_size, img_dim]);
            dim(1) = larger_size; dim(2) = larger_size;
        end
        % PCA projection
        pca_image = bsxfun(@minus, cur_image, expX) * eigVec(:,1:pca_lowdim);
        % linear stretch;
        pca_std = std(pca_image);
        pca_mean = mean(pca_image);
        stre_image =  bsxfun(@rdivide, bsxfun(@minus, pca_image, pca_mean), pca_std );
        stre_image = (stre_image + 1)* 128;
        stre_image(stre_image > 255) = 255;
        stre_image(stre_image < 0) = 0;
        stre_image = reshape(stre_image, [dim(1), dim(2), 3]);
        stre_image = permute(stre_image, [2,1,3]);
        % write the bmp image;
        if( ~ exist(cur_outpath, 'dir') )
            mkdir(cur_outpath);
        end
        imwrite(uint8(stre_image), cur_outpathname, outsusp); 
        count = count + 1;
    end
    fprintf('\nProcessing %.1f%%\n', ic * 100 / class_num);
end


rmpath('feature_ext');

% % pca transform for each image;
% count  = 1;
% for ic = 1 : class_num,
%     cur_path = class_path{ic};
%     cur_classname = class_name{ic};
%     for ii = 1 : class_imgnum(ic),
%         samp_name = image_name{count};
%         [~,name_split] = fileparts(samp_name);
%         samp_image_name = fullfile(cur_path, name_split);
%         cur_outpath = fullfile(out_image_path, cur_classname);
%         cur_outpathname = fullfile(cur_outpath, [name_split, '.', outsusp]);
%         % read image
%         [cur_image, dim] = freadenvi(samp_image_name);
%         cur_image = reshape(cur_image, dim);
%         % pca transformation;
%         pca_image = PCA1(cur_image, pca_lowdim);
%         % linear stretch;
%         numpxl = dim(1)*dim(2);
%         pca_image = reshape(pca_image, [numpxl, pca_lowdim]);
%         pca_std = std(pca_image);
%         pca_mean = mean(pca_image);
%         stre_image =  bsxfun(@rdivide, bsxfun(@minus, pca_image, pca_mean), pca_std );
%         stre_image = (stre_image + 1)* 128;
%         stre_image(stre_image > 255) = 255;
%         stre_image(stre_image < 0) = 0;
%         stre_image = reshape(stre_image, [dim(1), dim(2), pca_lowdim]);
%         % write the bmp image;
%         if( ~ exist(cur_outpath, 'dir') )
%             mkdir(cur_outpath);
%         end
%         imwrite(uint8(stre_image), cur_outpathname, outsusp); 
%         count = count + 1;
%     end
%     fprintf('\nProcessing %.1f%%\n', ic * 100 / class_num);
% end


