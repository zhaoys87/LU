function  DataInfo = HKWVImageSet2Mat(index_dir, image_dir, token, image_mean,  crop_shape)
%% DataInfo = HKWVImageSet2Mat(in_dir, image_dir, token,  crop_shape, image_mean)
% Read the image dataset into the mat format. The image in the directory should have the same color
%   format, and the same file format.
%input parameter:
%   index_dir :  input directory containing scene class folder in which the scene images locate
%   image_dir :  input directory of sample images
%   token  :  the suffix of the images file, default '*.tif'.
%   crop_shape : the constraint shape (width x height) by crop operation
%   image_mean : the image mean (imh * imw * imb);
%output parameter:
%   DataInfo;
%     DataInfo.class_num        : number of classes
%     DataInfo.class_name       : name of each class
%     DataInfo.class_path       : the path of each class
%     DataInfo.img_name         : contain the pathes for each image of each class
%     DataInfo.class_imgnum     : number of images contained in each class
%     DataInfo.img_label        : label of each image contained in img_name
%     DataInfo.fea_dim          : the dimension of feature 
%     DataInfo.mat_path         : the pathes of data matrix corresponding
%       to different classes; [width, height, band, imagenum ]
%     DataInfo.mat_label         : the pathes of data matrix corresponding to different classes;
%% Writed by Bei Zhao on Jun. 14, 2016.

disp('begin to preprocessing the image');
if(~exist('token','var'))
    token = '*.tif';
end
if(~exist('crop_shape','var'))
    crop_shape = [227, 227];
end

%% get the filelist
DataInfo = SearchFolder2Big(index_dir, token );
img_label = DataInfo.img_label;
labelset = unique(img_label);
class_num = DataInfo.class_num;

[patht, imagename] = fileparts(DataInfo.img_name{1});
img_path1 = fullfile(image_dir, imagename);
[~, pdim] = freadenvi(img_path1);
dim_feat = size(image_mean, 3);

%% get image data
DataInfo.mat_path = cell(class_num,1);
mat_label = zeros(1, sum(DataInfo.class_imgnum));
for ic = 1:class_num ,
	imglbl = labelset(ic);
	ids = find(img_label == imglbl);
	file_num = numel(ids);
	ind_set = DataInfo;
	CropArgData = zeros(crop_shape(1), crop_shape(2), dim_feat, file_num, 'single');
	for ii = 1:file_num,
        	[patht, img_name] = fileparts(ind_set.img_name{ids(ii)});
        	img_path = fullfile(image_dir, img_name);
        	[IW, pdim] = freadenvi(img_path);
        	IW = reshape(IW, pdim);
        	CropArgData(:,:,:,ii) = imresize(single(IW(:,:,1:dim_feat)-image_mean), crop_shape);
	end
	class_path = ind_set.class_path{ic};
	[dataset_path, class_name] = fileparts(class_path);
	mat_path = [dataset_path, '_mat']; 
	if (~exist(mat_path, 'dir'))
        	mkdir(mat_path);
	end
	mat_path_cur = fullfile(mat_path, [class_name, '.mat']);
	saveVariable( mat_path_cur,'CropArgData', CropArgData);

	DataInfo.mat_path{ic} = mat_path_cur;
	mat_label(:,ids) = imglbl;
	fprintf('Processing : %d%%, %s\n', int32(ic*100 /class_num), class_path );
end %ic
DataInfo.mat_label = mat_label(:);
DataInfo.fea_dim = dim_feat;
save([mat_path, '_info.mat'], 'DataInfo');
end

function saveVariable(filePathName, varName, CropArgData)
    save(filePathName, varName);
end

