% function Urban_Gen_GT_Index()
%% Urban_Gen_GT_Index is to generate the GT image index
% input:
%   block_image_path : the image stores the block ids;
%   ust_samp_path : the centers of the sampling images
%   large_image_path : the path of the large images;
% output:
%   out_image_path : the path of the output image
%% Write by Bei Zhao, zhaoys@cuhk.edu.hk. Jun. 10, 2016.

dataset = 4 ;
shuffle = false;
outputtxtfile = true; % true;
train_test = 0; % 1 training and 0 testing

if(dataset == 1)
    image_path = '../Scene_HSRI/HK-WV3/Block_Sampling_Images_C';
    out_path = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/HK-WV3/Sample_Images';
    if(train_test)
        Image_index = '../Scene_HSRI/HK-WV3/Training_Samples';
        out_txt = '../Scene_HSRI/HK-WV3/training_image_samples.txt';
    else
        Image_index = '../Scene_HSRI/HK-WV3/Testing_Samples';
        out_txt = '../Scene_HSRI/HK-WV3/test_image_samples.txt';
    end
elseif (dataset == 2)
    image_path = '../Scene_HSRI/HK-WV3-Shatin/Sample_Images';
    out_path = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/HK-WV3-Shatin/Sample_Images';
    if(train_test)
        Image_index = '../Scene_HSRI/HK-WV3-Shatin/Training_Samples';
        out_txt = '../Scene_HSRI/HK-WV3-Shatin/train_image_samples_shatin.txt';
    else
        Image_index = '../Scene_HSRI/HK-WV3-Shatin/Testing_Samples';
        out_txt = '../Scene_HSRI/HK-WV3-Shatin/test_image_samples_shatin.txt';
    end
elseif (dataset == 3)
    image_path = '../Scene_HSRI/HK-WV3/Block_Sampling_Images_C';
    out_path = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/HK-WV3/Social_Sample_Images';
    if(train_test)
        Image_index = '../Scene_HSRI/HK-WV3/Social_Training_Samples';
        shuffle = true;
        out_txt = '../Scene_HSRI/HK-WV3/social_hsr_tr_samples.txt';
        out_soc_sample_path = '../Scene_HSRI/HK-WV3/social_soc_tr_samples.txt';
    else
        Image_index = '../Scene_HSRI/HK-WV3/Social_Testing_Samples';    
        shuffle = false;
        out_txt = '../Scene_HSRI/HK-WV3/social_hsr_te_samples.txt';
        out_soc_sample_path = '../Scene_HSRI/HK-WV3/social_soc_te_samples.txt';
    end
    social_data_path = '../Scene_HSRI/HK-WV3/Social_Data/social_data.mat';
    out_socialimg_path = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/HK-WV3/Social_Data';
elseif (dataset == 4)
    image_path = '../Scene_HSRI/SZ-WorldView3/Block_Sampling_Images_C';
    out_path = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/SZ-WorldView3/Social_Sample_Images';
    if(train_test)
        Image_index = '../Scene_HSRI/SZ-WorldView3/Social_Training_Samples';
        shuffle = true;
        out_txt = '../Scene_HSRI/SZ-WorldView3/social_hsr_tr_samples.txt';
        out_soc_sample_path = '../Scene_HSRI/SZ-WorldView3/social_soc_tr_samples.txt';
    else
        Image_index = '../Scene_HSRI/SZ-WorldView3/Social_Testing_Samples';
        shuffle = false;
        out_txt = '../Scene_HSRI/SZ-WorldView3/social_hsr_te_samples.txt';
        out_soc_sample_path = '../Scene_HSRI/SZ-WorldView3/social_soc_te_samples.txt';
    end

    social_data_path = '../Scene_HSRI/SZ-WorldView3/Social_Data/sz_social_data.mat';
    out_socialimg_path = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/SZ-WorldView3/Social_Data';
else
    disp('the dataset should be in the defined range');
    return;
end
token = '*.jpg';

addpath('feature_ext');
DataInfo = SearchFolder2Big( Image_index, token );
rmpath('feature_ext');
image_name = DataInfo.img_name;
class_path = DataInfo.class_path;
class_imgnum = DataInfo.class_imgnum;
class_num = numel(class_imgnum);
numimage = sum(class_imgnum);

%% samples confirm
count = 1;
flag = zeros(numimage, 1);
for ic = 1 : class_num,
    cur_path = class_path{ic};
    for ii = 1 : class_imgnum(ic),
        samp_name = image_name{count};
        samp_pn = fullfile(cur_path, samp_name);
        [~,name_split] = fileparts(samp_name);
        samp_jpg_name = fullfile(image_path, name_split);
        if(exist(samp_jpg_name, 'file')),
            movefile(fullfile(image_path, name_split), fullfile(out_path, name_split));
            movefile(fullfile(image_path, [name_split,'.hdr']), fullfile(out_path, [name_split,'.hdr']));
            flag(count) = 1;
        else
            fprintf('The file %s in Class %d doesnot exist.\n', name_split, ic);
        end
        count = count + 1;
    end
    fprintf('\nProcessing %.1f%%\n', ic * 100 / class_num);
end

%% generate the image index;
count = 1;
flag = zeros(numimage, 1);
% get the strings
str_lists =  cell(numimage, 1);
str_count = 0;
for ic = 1 : class_num,
    cur_path = class_path{ic};
    for ii = 1 : class_imgnum(ic),
        samp_name = image_name{count};
        [~,name_split] = fileparts(samp_name);
        samp_image_name = fullfile(out_path, name_split);
        if(exist(samp_image_name, 'file') && exist([samp_image_name '.hdr'], 'file') ),
            str_count = str_count + 1;
            str_lists{str_count} = sprintf('%s %i',samp_image_name,ic-1);
        else
            fprintf('The file doesnot exist in class %i: %s\n', ic, samp_image_name);
        end
        count = count + 1;
    end
    fprintf('\nProcessing %.1f%%\n', ic * 100 / class_num);
end

% shuffle the order of strings
if(shuffle)
    orderlst = randperm(str_count);
else
    orderlst = 1 : str_count;
end

% write the strings
if(outputtxtfile),
    fid = fopen(out_txt,'w');
    if fid == -1
       fprintf('Cannot open the image %s\n', out_txt);
    end
    for ii=1 : str_count,
        ind = orderlst(ii);
        fprintf(fid,'%s \n',str_lists{ind});
    end
    fclose(fid);
end

%% generate the social image data with the same order as the HSR image data
disp('Generating social image with the same name and order of the HSR images');
socdat = load(social_data_path);
social_data = log(socdat.social_data + 1);
social_max = max(social_data(:));
social_min =  min(social_data(:));
social_data = uint8((social_data - social_min)*(255 / (social_max-social_min))); 
% should be revised by the ((x - mean) * 128 / std * 2 + 128)
dim_vec = size(social_data, 2);

if(outputtxtfile),
    fid = fopen(out_soc_sample_path,'w');
    if fid == -1
       fprintf('Cannot open the image %s\n', out_soc_sample_path);
    end
end

for ii=1 : str_count,
    ind = orderlst(ii);
    filepathname = str_lists{ind};
    [~, fn] = fileparts(filepathname);
    block_id = str2double(fn(1:4));
    if(block_id == 0)
        cur_vec = zeros(1, dim_vec, 'uint8');
    else
        cur_vec = social_data(block_id, :);
    end
    % write social image;    
    cur_name = strtok(fn);
    cur_outfile = fullfile(out_socialimg_path, cur_name);
    enviwrite(cur_vec, size(cur_vec, 2), size(cur_vec, 1) , 1,cur_outfile);
    % write txt file
    if(outputtxtfile),
        fprintf(fid,'%s \n',fullfile(out_socialimg_path, fn));
    end

    fprintf('\nProcessing %.1f%%\n', ii * 100 / str_count);
end

if(outputtxtfile),
    fclose(fid);
end


% end
