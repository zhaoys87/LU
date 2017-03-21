% function Urban_Compute_Mean()
%% Urban_Compute_Mean is to compute the mean image of the training images.
% input:
%   tr_jpg_path : the file path of the training images with jpg format;
%   image_path : the path of all training and testing samples with envi format; 
% output:
%   meanfile : the path of the mean image
%% Write by Bei Zhao, zhaoys@cuhk.edu.hk. Jun. 13, 2016.
dataset = 6; % 1: hk-wv3-Kawloon; 2 :1hk-wv3-shatin; 3 : social_data_hk; 4 : wuhan;
        % 5 : shenzhen-worldview3 ; 6 : social_data_shenzhen;
if(dataset == 1),
    tr_jpg_path = '../Scene_HSRI/HK-WV3/Training_Samples';
    image_path = '../Scene_HSRI/HK-WV3/Sample_Images';
    meanfile = '../Scene_HSRI/HK-WV3/hk-wv3-samples_mean';
    SSize = [256, 256];
    token = '*.jpg';
    WV3Dim = 8;
elseif(dataset == 2)
    tr_jpg_path = '../Scene_HSRI/HK-WV3-Shatin/Training_Samples';
    image_path = '../Scene_HSRI/HK-WV3-Shatin/Sample_Images';
    meanfile = '../Scene_HSRI/HK-WV3-Shatin/shatin-samples-mean';
    SSize = [256, 256];
    token = '*.jpg';
    WV3Dim = 8;
elseif(dataset == 3)
    tr_jpg_path = '../Scene_HSRI/HK-WV3/Social_Training_Samples';
    image_path = '../Scene_HSRI/HK-WV3/Social_Data';
    meanfile = '../Scene_HSRI/HK-WV3/social-data-mean';
    token = '*.jpg';
    SSize = [48, 1];
    WV3Dim = 1;
elseif(dataset == 4)
    tr_jpg_path = '../Scene_HSRI/Wuhan-IKONOS/ImageSamplesPCA';
    meanfile = 'wuhan/wuhan-mean';
    token = '*.bmp';
    SSize = [256, 256];
    WV3Dim = 3;
elseif(dataset == 5)
    tr_jpg_path = '../Scene_HSRI/SZ-WorldView3/Social_Training_Samples';
    image_path = '../Scene_HSRI/SZ-WorldView3/Social_Sample_Images';
    meanfile = '../Scene_HSRI/SZ-WorldView3/sz-samples-mean';
    SSize = [256, 256];
    token = '*.jpg';
    WV3Dim = 4;
elseif(dataset == 6)
    tr_jpg_path = '../Scene_HSRI/SZ-WorldView3/Social_Training_Samples';
    image_path = '../Scene_HSRI/SZ-WorldView3/Social_Data';
    meanfile = '../Scene_HSRI/SZ-WorldView3/sz-social-data-mean';
    token = '*.jpg';
    SSize = [48, 1];
    WV3Dim = 1;
else
    disp('the dataset type value should be in the defined range.')
    return;
end

addpath('feature_ext');
DataInfo = SearchFolder2Big( tr_jpg_path, token );
rmpath('feature_ext');

image_name = DataInfo.img_name;
class_path = DataInfo.class_path;
class_imgnum = DataInfo.class_imgnum;
class_num = numel(class_imgnum);
numimage = sum(class_imgnum);
count = 1;
mean_data = zeros(SSize(1), SSize(2), WV3Dim, 'single');
for ic = 1 : class_num,
    cur_path = class_path{ic};
    for ii = 1 : class_imgnum(ic),
        samp_name = image_name{count};
        if(dataset == 4)
            samp_image_pn = fullfile(cur_path, samp_name);
            curimage = single(imread(samp_image_pn));
        else
            [~,name_split] = fileparts(samp_name);
            samp_image_pn = fullfile(image_path, name_split);
            [curimage, p] = freadenvi(samp_image_pn);
            curimage = reshape(curimage,p);
        end
        mean_data = mean_data + curimage;
        count = count + 1;
    end
    fprintf('\nProcessing %.1f%%\n', ic * 100 / class_num);
end
mean_data = mean_data / numimage;
save([meanfile, '.mat'], 'mean_data');

if exist('caffe-master/matlab/+caffe', 'dir')
  addpath('caffe-master/matlab');
else
  error('Please run this demo from deep_learning/LU');
end

caf_io = caffe.io();
caf_io.write_mean(mean_data, [meanfile, '.binaryproto']);

%% test;
mean_test = caf_io.read_mean([meanfile, '.binaryproto']);
mean_test = mean_test - mean_data;
if ( sum(mean_test(:)) == 0 )
    disp('Success to write the binaryproto.');
else
    disp('Fail to write the binaryproto.');
end
% end
