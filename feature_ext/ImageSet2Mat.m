function  DataInfo = ImageSet2Mat(in_dir, token,  crop_shape, argu_type, image_mean)
%% DataInfo = ImageSet2Mat(in_dir, token,  crop_shape, argu_type, image_mean)
% Read the image dataset into the mat format. The image in the directory should have the same color
%   format, and the same file format.
%input parameter:
%   in_dir :  input directory containing scene class folder in which the scene images locate
%   token  :  the suffix of the images file, default '*.tif'.
%   crop_shape : the constraint shape (width x height) by crop operation
%   image_mean : the image mean (imh * imw * imb);
%   argu_type : the type of argument, 0, no argument (default); 1, corner
%       argumentation (5 times); 2, corner and flip argumenation (10 times).
%output parameter:
%   DataInfo;
%     DataInfo.class_num        : number of classes
%     DataInfo.class_name       : name of each class
%     DataInfo.class_path       : the path of each class
%     DataInfo.img_name         : contain the pathes for each image of each class
%     DataInfo.class_imgnum     : number of images contained in each class
%     DataInfo.img_label        : label of each image contained in img_name
%     DataInfo.fea_min         : the minimum value of feature
%     DataInfo.fea_max         : the maximum value of  feature 
%     DataInfo.fea_dim          : the dimension of feature 
%     DataInfo.fea_mean         : the mean of features 
%     DataInfo.mat_path         : the pathes of data matrix corresponding
%       to different classes; [width, height, band, imagenum ]
%     DataInfo.mat_label         : the pathes of data matrix corresponding to different classes;
%% Writed by Bei Zhao on Mar. 23, 2016.

disp('begin to preprocessing the image');
if(~exist('token','var'))
    token = '*.tif';
end
if(~exist('argu_type','var'))
    argu_type = 0;
end
if(~exist('crop_shape','var'))
    crop_shape = [224, 224];
end

if(~exist('image_mean','var'))
    bexist_mean = 0;
else
    bexist_mean = 1;
end

%% get the filelist
DataInfo = SearchFolder2Big(in_dir, token );
img_label = DataInfo.img_label;
labelset = unique(img_label);
class_num = DataInfo.class_num;

img_path1 = fullfile(DataInfo.class_path{1}, DataInfo.img_name{1});
if(strcmp(token, '*.hdr')),
    [~, pdim] = freadenvi(img_path1);
    image_channels = pdim(3);
else
    IW = imread(img_path1);
    image_channels = size(IW, 3); 
end

%% get image data
if(bexist_mean)
    [~,~,mean_band] = size(image_mean);
end

if (argu_type == 0),
	num_mul = 1;
elseif(argu_type == 1) 
	num_mul = 5; 
else % argu_type ==  2
    num_mul = 10;
end

fea_min = zeros(image_channels,1);
fea_min(:) = 255000;
fea_max = zeros(image_channels,1);
dim_feat = image_channels;
image_mean_cur = zeros( crop_shape(1), crop_shape(2), image_channels );

DataInfo.mat_path = cell(class_num,1);
mat_label = zeros(num_mul, sum(DataInfo.class_imgnum));
DataInfo.image_mean = image_mean_cur;

for ic = 1:class_num ,
	ids = find(img_label == labelset(ic));
	file_num = numel(ids);
	ind_set = DataInfo;
	imglbl = labelset(ic);
	class_path = ind_set.class_path{imglbl};
	CropArgData = zeros(crop_shape(1), crop_shape(2), image_channels, file_num * num_mul, 'single');
	Data_sid = 1;
	for ii = 1:file_num,
        % read image
        img_name = ind_set.img_name{ids(ii)};
        img_path = fullfile(class_path, img_name);
        if (strcmp(token, '*.hdr')),
            [IW, pdim] = freadenvi(img_path);
            IW = reshape(IW, pdim);
        else
            IW = imread(img_path);
            IW = IW(:, :, [3, 2, 1]); % convert from RGB to BGR
            IW = permute(IW, [2, 1, 3]); % permute width and height
        end
        
        if(bexist_mean == 0)
            if(mean_band ~= image_channels);
                CropArgData(:,:,:,Data_sid:Data_sid+num_mul-1) = ImageCropArgu(IW, crop_shape, argu_type);
            end
        else
    		CropArgData(:,:,:,Data_sid:Data_sid+num_mul-1) = ImageCropArgu(IW, crop_shape, argu_type, image_mean);
        end
        
        Data_sid = Data_sid + num_mul;
%         figure(201601);
%         for ifig = 1 : 5,
%             subplot(2, 5, ifig); imshow(uint8(CropArgData(:,:, :, ifig)));
%             subplot(2, 5, ifig + 5); imshow(uint8(CropArgData(:,:, :, ifig+5)));
%         end
	end %ii
	image_mean_cur =  sum(CropArgData, 4) / num_mul;
	cur_min = min(min(min(CropArgData,[],1),[],2),[],4);
	cur_max = max(max(max(CropArgData,[],1),[],2),[],4);
	fea_min = min(cur_min(:), fea_min );
	fea_max = max(cur_max(:), fea_max );

	[dataset_path, class_name] = fileparts(class_path);
	mat_path = [dataset_path, '_',num2str(crop_shape(1)), 'crop_', num2str(num_mul), 'times_mat']; 
	if (~exist(mat_path, 'dir'))
        mkdir(mat_path);
	end
	mat_path_cur = fullfile(mat_path, [class_name, '.mat']);
	saveVariable( mat_path_cur,'CropArgData', CropArgData);

	DataInfo.mat_path{ic} = mat_path_cur;
	mat_label(:,ids) = imglbl;
	DataInfo.image_mean = image_mean_cur +  DataInfo.image_mean;

	imshow(uint8(image_mean_cur / file_num ));
	fprintf('Processing : %d%%, %s\n', int32(ic*100 /class_num), class_path );
end %ic

DataInfo.mat_label = mat_label(:);
DataInfo.image_mean = DataInfo.image_mean / numel(img_label) ;
DataInfo.fea_min = fea_min;
DataInfo.fea_max = fea_max;
DataInfo.fea_dim = dim_feat;
save([mat_path, '_info'], 'DataInfo');
end

function saveVariable(filePathName, varName, CropArgData)
    save(filePathName, varName);
end

