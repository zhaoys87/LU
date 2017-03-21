clc; 
%% urban_anno_dcnn_hkwv3
% Demonstration for land-use annotation
%% bei zhao, zhaoys@cuhk.edu.hk, March 21, 2016
%% input data

crop_shape = [227, 227];
type = 4;
Sample_Type = 2; % type of sampling. 1 : unified sampling; 2 : street block sampling;
dataset = 1; % 1: hk-wv3-kawloon; 2: Shatin
iternum = 10000;

%     test_ft_proto = ['hk-wv3/transfer_ft/ft', num2str(type), '_alex_test.prototxt'];
%     weights_ft_file = ['hk-wv3/transfer_ft/results/ft', num2str(type), '_alexnet_iter_', ...
%         num2str(iternum), '.caffemodel'];
%     mean_file = 'hk-wv3/hk-wv3-samples_mean.mat';
    test_ft_proto = ['shatin-wv3/transfer_ft/ft', num2str(type), '_alex_test.prototxt'];
    weights_ft_file = ['shatin-wv3/transfer_ft/results/ft', num2str(type), '_alexnet_iter_', ...
        num2str(iternum), '.caffemodel'];
    mean_file = 'shatin-wv3/shatin-samples-mean.mat';

if(dataset==1)
    block_image_path = '../Scene_HSRI/HK-WV3/streetblock_subset';
    large_image_path = '../Scene_HSRI/HK-WV3/hk-wv3-subset';
    if(Sample_Type == 1)
        ust_samp_path = '../Scene_HSRI/HK-WV3/ust_samp_uni_ids';
    else
        ust_samp_path = '../Scene_HSRI/HK-WV3/ust_samp_block_ids_mat.mat';
    end
%   output_image = ['hk-wv3/transfer_ft/results/hk-wv3-cls-ver', num2str(type), '_samp', num2str(Sample_Type)];
    output_image = ['shatin-wv3/transfer_ft/results/hk-wv3-cls-ver', num2str(type), '_samp', num2str(Sample_Type)];
else
    block_image_path = '../Scene_HSRI/HK-WV3-Shatin/shatin_streetblock';
    large_image_path = '../Scene_HSRI/HK-WV3-Shatin/shatin_wv3';
    if(Sample_Type == 1)
        ust_samp_path = '../Scene_HSRI/HK-WV3-Shatin/shatin_samp_uni_ids';
    else
        ust_samp_path = '../Scene_HSRI/HK-WV3-Shatin/shatin_samp_block_ids_mat.mat';
    end
%   output_image = ['hk-wv3/transfer_ft/results/shatin-cls-ver', num2str(type), '_samp', num2str(Sample_Type)];
    output_image = ['shatin-wv3/transfer_ft/results/shatin-cls-ver', num2str(type), '_samp', num2str(Sample_Type)];
end



Marker_Value = 9999;
SSize = 256;    % sample size;
AreaS = SSize * SSize;
semiSSize = floor(SSize / 2);
class_num = 11;

WVDim = 8;
water_class = 7;
WVPrec = 'uint16';
NDWI_thd = 0.6;
NDWI_area_thd = 0.9 * AreaS;
Coastline = 1;
NIR2 = 8;
RGB_Band = [8,5,3];
Band_Reorder = [5,3,2,1,4,6,7,8];
if(type == 1)
model_channel = 3;
else
model_channel = 8;
end

gpu_id  = 0;

if exist('caffe-master/matlab/+caffe', 'dir')
  addpath('caffe-master/matlab');
else
  error('Please run this demo from deep_learning/LU');
end
addpath('feature_ext');
tic;

% read the ust and blocks sampling ids;
tic;
[block_image, dim] = freadenvi( block_image_path );
block_image = reshape(uint16(block_image), dim);

if(Sample_Type == 1)
    [samp_ids, dim] = freadenvi( ust_samp_path );
    samp_ids = reshape(uint16(samp_ids), dim);
    [sidxs, sidys] = find( samp_ids == Marker_Value );
    block_ids = block_image(sidxs + sidys*dim(1));
else
    data = load(ust_samp_path);
    sidxs = data.samp_ids(:,1);
    sidys = data.samp_ids(:,2);
    block_ids = data.samp_ids(:,3);
end
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

% fid = fopen(large_image_path);
% large_image = uint16( fread(fid,[numpxls, dim(3)], prec) );
% fclose(fid);
% 
% NDWI = single( large_image(:,Coastline)) - single( large_image(:,NIR2) );
% NDWI = NDWI ./ single( large_image(:,Coastline) + large_image(:,NIR2) ) ;
% NDWI = NDWI > NDWI_thd;
% NDWI = reshape(NDWI, [dim(1), dim(2)]);
% 
% large_image_c = zeros(numpxls, dim(3), 'uint8');
% mean_v = mean(large_image);
% std2_v = std(single(large_image)) * 2;
% scale_v = single(mean_v) + std2_v;
% blockpxl = floor(numpxls/2);
% for ib = 1 : dim(3),
%     norm_image =  single(large_image(1:blockpxl,ib)) * 255 / scale_v(ib);
%     large_image_c(1:blockpxl,ib) = uint8(norm_image);
% end
% for ib = 1 : dim(3),
%     norm_image =  single(large_image(1+blockpxl:end,ib)) * 255 / scale_v(ib);
%     large_image_c(1+blockpxl:end,ib) = uint8(norm_image);
% end
% large_image_c = reshape(large_image_c, dim);
% 
% clear norm_image large_image;
% save([large_image_path, '_c.mat'], 'large_image_c', 'NDWI');

% % 2 : read large image 
load([large_image_path, '_c.mat']);
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

    NDWI_cur = NDWI( sidxs_min(iid) : sidxs_max(iid), ...
        sidys_min(iid) : sidys_max(iid));
    if( sum(NDWI_cur(:)) > NDWI_area_thd),
        cur_pro = anno_scene_pro(sidxs_min(iid) : sidxs_max(iid),...
                sidys_min(iid) : sidys_max(iid), water_class);
        cur_pro(block_pxls) = cur_pro(block_pxls) + 1;
        anno_scene_pro(sidxs_min(iid) : sidxs_max(iid),...
                sidys_min(iid) : sidys_max(iid), water_class) = cur_pro;
        cur_ids = anno_scene_cnt(sidxs_min(iid) : sidxs_max(iid),...
                sidys_min(iid) : sidys_max(iid));
        cur_ids(block_pxls) = cur_ids(block_pxls) +1;
        anno_scene_cnt(sidxs_min(iid) : sidxs_max(iid),...
                sidys_min(iid) : sidys_max(iid)) = cur_ids;
        continue;
    end
    
    samp_images = large_image_c( sidxs_min(iid) : sidxs_max(iid), ...
        sidys_min(iid) : sidys_max(iid), :);
    samp_images = samp_images(:,:,Band_Reorder);
    samp_images =  single(samp_images(:,:,1:mean_size(3)))- mean_data ;
    cropdata = samp_images(CenW:CenW+crop_shape(1)-1, CenH:CenH+crop_shape(2)-1, 1:model_channel);
    net.forward({cropdata});
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

if(Sample_Type == 2),
    disp('Restrict label with the street block');
    tic;
    blockidmax = max(block_image(:));
    num_block = blockidmax + 1;
    block_pxl_pro = zeros(class_num,num_block);
    block_pxl_count = zeros(num_block, 1);
    for irow = 1 : dim(1),
        for icol = 1 : dim(2),
            block_id = block_image(irow, icol) + 1;
            block_pxl_pro(:, block_id) = block_pxl_pro(:, block_id) + ...
                reshape( anno_scene_pro(irow, icol, :),  [class_num, 1]);
            block_pxl_count(block_id) = block_pxl_count(block_id) + 1;
        end
    end
    block_pxl_pro = block_pxl_pro ./ repmat(block_pxl_count', [class_num, 1]);

    disp('Generate the labels...');
    [~, block_labels] = max(block_pxl_pro,[],1);
    rev_labels = zeros(dim(1),dim(2),'uint8');
    rev_pro = zeros(dim(1),dim(2), class_num, 'single');
    for irow = 1 : dim(1),
        for icol = 1 : dim(2),
            block_id = block_image(irow, icol) + 1;
            rev_labels(irow, icol) = block_labels(block_id);
            rev_pro(irow, icol,:) = block_pxl_pro(:, block_id);
        end
    end
    toc;
    total_time = toc;
    fprintf('The total cost time (s) : %.1f\n', total_time);
    enviwrite(rev_labels, dim(1), dim(2), 1, [output_image '_rev']);
    enviwrite(rev_pro, dim(1), dim(2), class_num, [output_image '_pro_rev']);
end

rmpath('caffe-master/matlab');
rmpath('feature_ext');