% function Urban_Gen_Samplings()
%% Urban_Gen_Samplings is to generate the samples from the street block images.
% input:
%   block_image_path : the image stores the block ids;
% output:
%   ust_samp_path : the centers of the sampling images
%% Write by Bei Zhao, zhaoys@cuhk.edu.hk. Jun. 7, 2016.
Sample_Type = 2; % 1 : unified sampling; 2: the sampling with street block;
dataset = 3; % 1: hk-wv3-Kawloon; 2 :1hk-wv3-shatin; 3 : shenzhen;
if(dataset == 1)
    block_image_path = '../Scene_HSRI/HK-WV3/streetblock_subset';
    if(Sample_Type == 1)
        ust_samp_path = '../Scene_HSRI/HK-WV3/ust_samp_uni_ids';
    else
        ust_samp_path = '../Scene_HSRI/HK-WV3/ust_samp_block_ids';
    end
elseif(dataset == 2),
    block_image_path = '../Scene_HSRI/HK-WV3-Shatin/shatin_streetblock';
    if(Sample_Type == 1)
        ust_samp_path = '../Scene_HSRI/HK-WV3-Shatin/shatin_samp_uni_ids';
    else
        ust_samp_path = '../Scene_HSRI/HK-WV3-Shatin/shatin_samp_block_ids';
    end
elseif(dataset == 3),
    block_image_path = '../Scene_HSRI/SZ-WorldView3/SZ_street_block';
    if(Sample_Type == 1)
        ust_samp_path = '../Scene_HSRI/SZ-WorldView3/sz_samp_uni_ids';
    else
        ust_samp_path = '../Scene_HSRI/SZ-WorldView3/sz_samp_block_ids';
    end
else
    fprintf('The id of the dataset should be within [1, %d]', 3);
    return;
end

SSize = 256;    % sample size;
semiSSize = floor(SSize / 2);
SSpacing = 128; % sample spacing;
num_storage = 8000;
mask_value = [1 num_storage] ;
%% generate the ground truth image

samp_ids = zeros(num_storage, 3);
[block_image, dim] = freadenvi(block_image_path);
block_image = uint16(reshape(block_image, dim));
ust_gt = zeros(dim(1), dim(2),'uint8');
if (Sample_Type == 1),
    img_remX = mod(dim(1) - SSize, SSpacing);
    img_offsetX = floor(img_remX/2)+1+semiSSize;
    img_remY = mod(dim(2) - SSize, SSpacing);
    img_offsetY = floor(img_remY/2)+1+semiSSize;
    [img_gridX,img_gridY] = meshgrid(img_offsetX : SSpacing: dim(1) - semiSSize+1,...
                                 img_offsetY : SSpacing: dim(2) - semiSSize+1);
    num_samples = numel(img_gridX);
    tot_samp_ids = img_gridX + (img_gridY-1)*dim(1);
    samples_block_ids = block_image(tot_samp_ids);
    sel_samp_ids_bool = (samples_block_ids >= mask_value(1)) & (samples_block_ids <= mask_value(2));
    sel_samp_ids = tot_samp_ids(sel_samp_ids_bool);
    ust_gt( sel_samp_ids ) = 1;
    
    num_sel_samples = numel(sel_samp_ids);
    samp_ids = [img_gridX(sel_samp_ids_bool) img_gridY(sel_samp_ids_bool) samples_block_ids(sel_samp_ids_bool) ];
else
    sel_block_flag = block_image >= mask_value(1) & block_image <= mask_value(2);
    block_ids = unique( block_image(sel_block_flag) );
    num_blocks = numel(block_ids);
    num_samples_block = 0;
    for ib = 1 : num_blocks,
        [xids, yids] = find( block_image == block_ids(ib) );
        [sxids, syids] = Urban_Block_Sampling2(xids, yids, SSize, SSpacing);
        % center should locate in the position [semiSSize+1 : dim(1)-semiSSize-1, ...
        %       semiSSize+1 : dim(2)-semiSSize-1 
        sxids((sxids < semiSSize + 1)) = semiSSize+1;
        sxids((sxids > dim(1)-semiSSize-1)) = dim(1)-semiSSize +1;
        syids((syids < semiSSize + 1)) = semiSSize+1;
        syids((syids > dim(2)-semiSSize-1)) = dim(2)-semiSSize + 1;
        % xids should be
        ust_gt( sxids + (syids-1)*dim(1)) = 1;
        cur_num_samples = numel(sxids);
        if (num_samples_block + cur_num_samples > num_storage),
            num_storage = num_storage + 8000;
            samp_ids_t = zeros(num_storage, 3);
            samp_ids_t(1:num_storage-8000,:) = samp_ids;
            samp_ids = samp_ids_t;
            clear 'samp_ids_t';
        end
        samp_ids(num_samples_block+1 : num_samples_block + cur_num_samples, 1) = sxids;
        samp_ids(num_samples_block+1 : num_samples_block + cur_num_samples, 2) = syids;
        samp_ids(num_samples_block+1 : num_samples_block + cur_num_samples, 3) = block_ids(ib);
        num_samples_block = num_samples_block + cur_num_samples;
        fprintf('Processing %.1f %%\n', ib * 100 / num_blocks);
    end
    samp_ids = samp_ids(1:num_samples_block, :);
end
save([ust_samp_path, '_mat.mat'], 'samp_ids');
enviwrite(ust_gt, dim(1), dim(2), 1, ust_samp_path);
clear img_remY img_remX img_offsetX img_offsetY;
% end
