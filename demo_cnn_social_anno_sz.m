clc; 
%% urban_anno_dcnn_sz
% Demonstration for land-use annotation
%% bei zhao, zhaoys@cuhk.edu.hk, Nov. 11, 2016
%% input data

crop_shape = [227, 227];
type = 4; % 12: (image_alex_fe + social_cnn) + (small_nets + social_cnn)
Sample_Type = 1; % type of sampling. 1 : unified sampling; 2 : street block sampling;
iternum = 10000;
class_num = 11;
read_large_flag = 1;  % 0: false; 1: true;

if(type>6)
    test_ft_proto = ['social-shenzhen/union/ft', num2str(type), '_alex_test.prototxt'];
    weights_ft_file = ['social-shenzhen/union/results/ft', num2str(type), '_alexnet_iter_', ...
        num2str(iternum(1)), '.caffemodel'];
    output_image = ['social-shenzhen/union/anno/sz_social_img_cnn_ver', num2str(type), '_samp', num2str(Sample_Type)];
elseif(type < 5)
    test_ft_proto = ['social-shenzhen/transfer_ft/ft', num2str(type), '_alex_test.prototxt'];
    weights_ft_file = ['social-shenzhen/transfer_ft/results/ft', num2str(type), '_alexnet_iter_', ...
        num2str(iternum(1)), '.caffemodel'];
    output_image = ['social-shenzhen/transfer_ft/anno/sz_img_cnn_ver', num2str(type), '_samp', num2str(Sample_Type)];
else
    test_ft_proto = ['social-shenzhen/social-only/', num2str(type), 'feat_socialdata.prototxt'];
    weights_ft_file = ['social-shenzhen/social-only/results/cnn_socialdata_', num2str(type), 'train_iter_', ...
        num2str(iternum(1)), '.caffemodel'];
    output_image = ['social-shenzhen/social-only/anno/sz_social_cnn_ver', num2str(type), '_samp', num2str(Sample_Type)];
end
mean_file = 'social-shenzhen/sz-samples-mean.mat';
mean_social_file = 'social-shenzhen/sz-social-data-mean.mat';

social_data_path = '../Scene_HSRI/SZ-WorldView3/Social_Data/sz_social_data.mat';
block_image_path = '../Scene_HSRI/SZ-WorldView3/SZ_street_block';
large_image_path = '../Scene_HSRI/SZ-WorldView3/SZ_WV3';
mask_path = '../Scene_HSRI/SZ-WorldView3/SZ_WV3_mask';
if(Sample_Type == 1)
    ust_samp_path = '../Scene_HSRI/SZ-WorldView3/sz_samp_uni_ids_mat.mat';
else
    ust_samp_path = '../Scene_HSRI/SZ-WorldView3/sz_samp_block_ids_mat.mat';
end
Band_Reorder = [3,2,1,4];

Marker_Value = 9999;
SSize = 256;    % sample size;
AreaS = SSize * SSize;
semiSSize = floor(SSize / 2);
WVDim = 4;
WVPrec = 'uint16';

[maskimage, ~] = freadenvi( mask_path );
maskimage = logical(maskimage);

if(type == 1 || type == 2)
    model_channel = 3;
else
    model_channel = 4;
end

gpu_id  = 0;

if exist('caffe-master/matlab/+caffe', 'dir')
  addpath('caffe-master/matlab');
else
  error('Please run this demo from deep_learning/LU');
end
addpath('feature_ext');

% processing the social data;
socdat = load(social_data_path);
social_data = log(socdat.social_data + 1);
social_max = max(social_data(:));
social_min =  min(social_data(:));
social_dim_vec = size(social_data, 2);
social_data = uint8((social_data' - social_min)*(255 / (social_max-social_min))); 
% should be revised. by sigma * 2 like image data.


meansocload = load(mean_social_file);
mean_social_data = meansocload.mean_data;
% read the ust and blocks sampling ids;
tic;
[block_image, dim] = freadenvi( block_image_path );
block_image = reshape(uint16(block_image), dim);

data = load(ust_samp_path);
sidxs = data.samp_ids(:,1);
sidys = data.samp_ids(:,2);
block_ids = data.samp_ids(:,3);

num_ids = numel(sidxs);
sidxs_min = sidxs - semiSSize;
sidys_min = sidys - semiSSize;
sidxs_max = sidxs + semiSSize -1;
sidys_max = sidys + semiSSize -1;

meandload = load(mean_file);
mean_data = meandload.mean_data;
mean_size = size(mean_data);

% 1: read large image and convert data type to uint8 type;
dim(3) = WVDim;
prec = WVPrec;
numpxls = dim(1)*dim(2);

if(read_large_flag == 0)
    fid = fopen(large_image_path);
    large_image = uint16( fread(fid,[numpxls, dim(3)], prec) );
    fclose(fid);

    large_image_c = zeros(numpxls, dim(3), 'uint8');
    mean_v = mean(large_image(maskimage,:));
    std2_v = std(single( large_image(maskimage,:) )) * 2;
    scale_v = single(mean_v) + std2_v;
    for ib = 1:dim(3)
        norm_image = single( large_image(maskimage,ib) ) * 255 / scale_v(ib);
        large_image_c(maskimage,ib) = uint8(norm_image);
    end
    large_image_c = reshape(large_image_c, dim);
    clear norm_image large_image;
    save([large_image_path, '_c.mat'], 'large_image_c');
else
    load([large_image_path, '_c.mat']);
end

toc;

% extract the split image, perform classification, and annotate the image.
tic;
anno_scene_pro = zeros(dim(1),dim(2), class_num,'single');
anno_scene_cnt = zeros(dim(1),dim(2),'single');

caffe.set_device(gpu_id);
net = caffe.Net(test_ft_proto, weights_ft_file, 'test'); % create net and load weights
CenW = floor((mean_size(1) - crop_shape(1) + 1)/2) ;
CenH = floor((mean_size(2) - crop_shape(2) + 1)/2) ;
for iid = 1 : num_ids,
    block_id = block_ids(iid);
    if (Sample_Type == 2),
        block_pxls = block_image(sidxs_min(iid) : sidxs_max(iid), ...
            sidys_min(iid) : sidys_max(iid)) == block_id;
    else
        block_pxls = true(SSize, SSize);
    end
   
    samp_images = large_image_c( sidxs_min(iid) : sidxs_max(iid), ...
        sidys_min(iid) : sidys_max(iid), :);
    samp_images = samp_images(:,:,Band_Reorder);
    samp_images =  single(samp_images(:,:,1:mean_size(3)))- mean_data ;
    cropdata = samp_images(CenW:CenW+crop_shape(1)-1, CenH:CenH+crop_shape(2)-1, 1:model_channel);
    
    inputdata = cell(2,1);
    inputdata{1} = cropdata;
    if(block_id)
        socialvalue = single(social_data(:,block_id)) - mean_social_data;
    else
        socialvalue = zeros(social_dim_vec,1,'single');
    end
    inputdata{2} = socialvalue;
    if(type < 5)
        net.forward(inputdata(1));
    elseif(type > 6)
        net.forward(inputdata);
    else
        net.forward(inputdata(2));
    end
    cur_pro =  net.blobs('prob').get_data();
    
    cur_pro = repmat(cur_pro', AreaS, 1);
    cur_pro = reshape(cur_pro, SSize, SSize, class_num);
    rem_pro = repmat(~block_pxls, 1, 1, class_num);
    cur_pro(rem_pro) = 0;
    anno_scene_pro(sidxs_min(iid) : sidxs_max(iid),...
            sidys_min(iid) : sidys_max(iid), :) = ...
    anno_scene_pro(sidxs_min(iid) : sidxs_max(iid),...
            sidys_min(iid) : sidys_max(iid), :) + cur_pro;
    cur_ids = single(block_pxls);
    anno_scene_cnt(sidxs_min(iid) : sidxs_max(iid),...
            sidys_min(iid) : sidys_max(iid)) = ...
        anno_scene_cnt(sidxs_min(iid) : sidxs_max(iid),...
            sidys_min(iid) : sidys_max(iid))  + cur_ids;
    
    if(mod(iid, 50) == 0)
        fprintf('Processing %.1f%%\n', iid * 100 / num_ids);
    end
end
caffe.reset_all();
toc;

process_ids = anno_scene_cnt > 0;
for ic = 1 : class_num,
    cur_bandpro = anno_scene_pro(:,:,ic);
    cur_bandpro(process_ids) = cur_bandpro(process_ids)./ anno_scene_cnt(process_ids);
    anno_scene_pro(:,:,ic) =  cur_bandpro;
end

[~, pred_labels] = max(anno_scene_pro, [], 3);
pred_labels = uint8(pred_labels);
pred_labels(~process_ids) = 0;
enviwrite(pred_labels, dim(1), dim(2), 1, output_image);

maskimage = reshape(maskimage,[dim(1), dim(2)]);
if(Sample_Type == 2),
    disp('Restrict label with the street block');
    tic;
    blockidmax = max(block_image(:));
    num_block = blockidmax;
    block_pxl_pro = zeros(class_num,num_block);
    block_pxl_count = zeros(num_block, 1);
    for irow = 1 : dim(1),
        for icol = 1 : dim(2),
            if(maskimage(irow, icol)),
                block_id = block_image(irow, icol) + 1;
                block_pxl_pro(:, block_id) = block_pxl_pro(:, block_id) + ...
                    reshape( anno_scene_pro(irow, icol, :),  [class_num, 1]);
                block_pxl_count(block_id) = block_pxl_count(block_id) + 1;
            end
        end
        if(mod(irow, 50) == 0)
            fprintf('Processing %.1f%%\n', irow * 100 / dim(1));
        end
    end
    proc_flag = block_pxl_count > 0;
    proc_flag = repmat(proc_flag', [class_num, 1]);
    block_pxl_count = repmat(block_pxl_count', [class_num, 1]);
    block_pxl_pro(proc_flag) = block_pxl_pro(proc_flag) ./ block_pxl_count(proc_flag);

    disp('Generate the labels...');
    [~, block_labels] = max(block_pxl_pro,[],1);
    rev_labels = zeros(dim(1),dim(2),'uint8');
    rev_pro = zeros(dim(1),dim(2), class_num, 'single');
    for irow = 1 : dim(1),
        for icol = 1 : dim(2),
            if(maskimage(irow, icol)),
                block_id = block_image(irow, icol) + 1;
                rev_labels(irow, icol) = block_labels(block_id);
                rev_pro(irow, icol,:) = block_pxl_pro(:, block_id);
            end
        end
        if(mod(irow, 50) == 0)
            fprintf('Processing %.1f%%\n', irow * 100 / dim(1));
        end
    end
    toc;
    total_time = toc;
    fprintf('The total cost time (s) : %.1f\n', total_time);
    enviwrite(rev_labels, dim(1), dim(2), 1, [output_image '_rev']);
    if(Sample_Type == 2)
        enviwrite(rev_pro, dim(1), dim(2), class_num, [output_image '_pro_rev']);
    end
end

rmpath('caffe-master/matlab');
rmpath('feature_ext');