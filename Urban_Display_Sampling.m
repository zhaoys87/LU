% function Urban_display_Samplings()
%% Urban_display_Samplings is to generate the samples from the street block images.
% input:
%   block_image_path : the image stores the block ids;
% output:
%   ust_samp_path : the centers of the sampling images
%% Write by Bei Zhao, zhaoys@cuhk.edu.hk. Jun. 7, 2016.
Sample_Type = 2; % 1 : unified sampling; 2: the sampling with street block;
block_image_path = '../Scene_HSRI/HK-WV3/SamplingStreetBlock/subset_images_street_block';
if(Sample_Type == 1)
    ust_samp_path = '../Scene_HSRI/HK-WV3/SamplingStreetBlock/subset_samp_uni_ids';
else
    ust_samp_path = '../Scene_HSRI/HK-WV3/SamplingStreetBlock/subset_samp_block_ids';
end
SSize = 256;    % sample size;
semiSSize = floor(SSize / 2);
SSpacing = 128; % sample spacing;
CenterSize = 9;
%% generate the ground truth image
[block_image, dim] = freadenvi(block_image_path);
block_image = uint16(reshape(block_image, dim));
ust_gt = false(dim(1), dim(2)) ;
skel_gt = false(dim(1), dim(2)) ;
if (Sample_Type == 1),
    img_remX = mod(dim(1) - SSize, SSpacing);
    img_offsetX = floor(img_remX/2)+1+semiSSize;
    img_remY = mod(dim(2) - SSize, SSpacing);
    img_offsetY = floor(img_remY/2)+1+semiSSize;
    [img_gridX,img_gridY] = meshgrid(img_offsetX : SSpacing: dim(1) - semiSSize+1,...
                                 img_offsetY : SSpacing: dim(2) - semiSSize+1);
    num_samples = numel(img_gridX);
    ust_gt( img_gridX + (img_gridY-1)*dim(1)) = true;
    ust_gt = imdilate(ust_gt, strel('disk', CenterSize)); 
    enviwrite(uint8(ust_gt), dim(1), dim(2), 1, ust_samp_path);

else
    block_ids = unique(block_image);
    num_blocks = numel(block_ids);
    for ib = 1 : num_blocks,
        [xids, yids] = find( block_image == block_ids(ib) );
        [sxids, syids, cur_ske] = Urban_Block_Sampling2(xids, yids, SSize, SSpacing);
        ust_gt( sxids + (syids-1)*dim(1)) = true;
        minxids = min(xids); maxxids = max(xids);
        minyids = min(yids); maxyids = max(yids);
        skel = false(dim(1), dim(2));
        skel(minxids:maxxids, minyids:maxyids) = cur_ske;
        skel_gt(skel) = true;
        fprintf('Processing %.1f %%\n', ib * 100 / num_blocks);
    end
    ust_gt = imdilate(ust_gt, strel('disk', CenterSize)); 
    skel_gt = imdilate(skel_gt, strel('disk', 1));

    enviwrite(uint8(ust_gt), dim(1), dim(2), 1, ust_samp_path);
    enviwrite(uint8(skel_gt), dim(1), dim(2), 1, [ust_samp_path '_skel'] );

end


% end
