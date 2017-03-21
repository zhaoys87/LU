% function Urban_Image_Split()
%% Urban_Image_Split is to split the large image into small images by the use of the street block images, and the sample ids.
% input:
%   block_image_path : the image stores the block ids;
%   ust_samp_path : the centers of the sampling images
%   large_image_path : the path of the large images;
% output:
%   out_image_path : the path of the output image
%% Write by Bei Zhao, zhaoys@cuhk.edu.hk. Jun. 7, 2016.

Sample_Type = 1; % 1 : unified sampling; 2: the sampling with street block;
Image_Type = 2; % 1 : original unsigned short image; 2: the stretched unsigned char image;
flag_jpg = 0; % 1 : output the jpg files; 0, do not output the jpg file;
dataset = 3; % 1 : hk-wv3-Kawloon; 2 :1hk-wv3-shatin;  3 : ShenZhen WV3
WVDim = 0;
if(dataset==1)
    block_image_path = '../Scene_HSRI/HK-WV3/streetblock_subset';
    large_image_path = '../Scene_HSRI/HK-WV3/hk-wv3-subset';
    if(Sample_Type == 1)
        ust_samp_path = '../Scene_HSRI/HK-WV3/ust_samp_uni_ids';
        if(Image_Type ==1)
            out_image_path = '../Scene_HSRI/HK-WV3/Uni_Sampling_Images';
        else
            out_image_path = '../Scene_HSRI/HK-WV3/Uni_Sampling_Images_C';
        end
    else
        ust_samp_path = '../Scene_HSRI/HK-WV3/ust_samp_block_ids';
        if(Image_Type ==1)
            out_image_path = '../Scene_HSRI/HK-WV3/Block_Sampling_Images';
        else
            out_image_path = '../Scene_HSRI/HK-WV3/Block_Sampling_Images_C';
        end
    end
    WVDim = 8;
    NDWI_area_thd = 0.9;
    Coastline = 1;
    NIR2 = 8;
    RGB_Band = [8,5,3];
    Band_Reorder = [5,3,2,1,4,6,7,8];
elseif(dataset==2)
    block_image_path = '../Scene_HSRI/HK-WV3-Shatin/shatin_streetblock';
    large_image_path = '../Scene_HSRI/HK-WV3-Shatin/shatin_wv3';
    if(Sample_Type == 1)
        ust_samp_path = '../Scene_HSRI/HK-WV3-Shatin/shatin_samp_uni_ids';
        if(Image_Type ==1)
            out_image_path = '../Scene_HSRI/HK-WV3-Shatin/Uni_Sampling_Images';
        else
            out_image_path = '../Scene_HSRI/HK-WV3-Shatin/Uni_Sampling_Images_C';
        end
    else
        ust_samp_path = '../Scene_HSRI/HK-WV3-Shatin/shatin_samp_block_ids';
        if(Image_Type ==1)
            out_image_path = '../Scene_HSRI/HK-WV3-Shatin/Block_Sampling_Images';
        else
            out_image_path = '../Scene_HSRI/HK-WV3-Shatin/Block_Sampling_Images_C';
        end
    end
    WVDim = 8;
    NDWI_area_thd = 0.9;
    Coastline = 1;
    NIR2 = 8;
    RGB_Band = [8,5,3];
    Band_Reorder = [5,3,2,1,4,6,7,8];
elseif(dataset==3)
    block_image_path = '../Scene_HSRI/SZ-WorldView3/SZ_street_block';
    large_image_path = '../Scene_HSRI/SZ-WorldView3/SZ_WV3';
    mask_path = '../Scene_HSRI/SZ-WorldView3/SZ_WV3_mask';
    if(Sample_Type == 1)
        ust_samp_path = '../Scene_HSRI/SZ-WorldView3/sz_samp_uni_ids_mat.mat';        
        if(Image_Type ==1)
            out_image_path = '../Scene_HSRI/SZ-WorldView3/Uni_Sampling_Images';
        else
            out_image_path = '../Scene_HSRI/SZ-WorldView3/Uni_Sampling_Images_C';
        end
    else
        ust_samp_path = '../Scene_HSRI/SZ-WorldView3/sz_samp_block_ids_mat.mat';
        if(Image_Type ==1)
            out_image_path = '../Scene_HSRI/SZ-WorldView3/Block_Sampling_Images';
        else
            out_image_path = '../Scene_HSRI/SZ-WorldView3/Block_Sampling_Images_C';
        end
    end
    WVDim = 4;
    NIR2 = 4;
    RGB_Band = [4,3,2];
    Band_Reorder = [3,2,1,4];
else
    fprintf('The id of the dataset should be within [1, %d]', 3);
    return;
end

if(~ exist('out_image_path', 'dir'))
    mkdir(out_image_path);
end

Marker_Value = 9999;
SSize = 256;    % sample size;
AreaS = SSize * SSize;
semiSSize = floor(SSize / 2);

WVPrec = 'uint16';
RGB_Scale = [255/500, 255/400, 255/400]; % donomiator are the (mean + 2* std) of the corresponding bands;
%% read the ust and blocks sampling ids;
dim = zeros(3,1);
if(dataset == 3)
    samp_ids_load = load(ust_samp_path);
    sidxs = samp_ids_load.samp_ids(:,1);
    sidys = samp_ids_load.samp_ids(:,2);
    [maskimage, dim] = freadenvi( mask_path );
    maskimage = logical(maskimage);
    blockids = samp_ids_load.samp_ids(:,3);
else
    [block_image, dim] = freadenvi( block_image_path );
    block_image = reshape(uint16(block_image), dim);
    samp_ids = freadenvi( ust_samp_path );
    samp_ids = reshape(uint16(samp_ids), [dim(1), dim(2)]);
    [sidxs, sidys] = find( samp_ids == Marker_Value );
    maskimage = true(dim(1)*dim(2),1);
end
num_ids = numel(sidxs);

sidxs_min = sidxs - semiSSize;
sidys_min = sidys - semiSSize;
sidxs_max = sidxs + semiSSize -1;
sidys_max = sidys + semiSSize -1;
%% read large image and write the splitted origin images;
dim(3) = WVDim;
prec = WVPrec;
fid = fopen(large_image_path);
numpxls = dim(1)*dim(2);
large_image = uint16( fread(fid,[numpxls, dim(3)], prec) );
fclose(fid);
if(Image_Type ==2),
    large_image_c = zeros(numpxls, dim(3), 'uint8');
    mean_v = mean(large_image(maskimage,:));
    std2_v = std(single( large_image(maskimage,:) )) * 2;
    scale_v = single(mean_v) + std2_v;
    if(dataset ==  3)
        for ib = 1:dim(3)
            norm_image = single( large_image(maskimage,ib) ) * 255 / scale_v(ib);
            large_image_c(maskimage,ib) = uint8(norm_image);
        end
    else
        blockpxl = floor(sel_numpxls/2);
        for ib = 1 : dim(3),
            norm_image =  single(large_image(1:blockpxl,ib)) * 255 / scale_v(ib);
            large_image_c(1:blockpxl,ib) = uint8(norm_image);
        end
        for ib = 1 : dim(3),
            norm_image =  single(large_image(1+blockpxl:end,ib)) * 255 / scale_v(ib);
            large_image_c(1+blockpxl:end,ib) = uint8(norm_image);
        end
    end
    large_image_c = reshape(large_image_c, dim);
    clear norm_image, 
end
large_image = reshape( large_image, dim );

for iid = 1 : num_ids,
    samp_x = sidxs(iid);
    samp_y = sidys(iid);
    samp_images = large_image( sidxs_min(iid) : sidxs_max(iid), ...
        sidys_min(iid) : sidys_max(iid), :);
    % skip the sampling image covering the water body
    if(dataset==3)
        block_id = blockids(iid);
    else
        NDWI = double( samp_images(:,:,Coastline)) - double( samp_images(:,:,NIR2) );
        NDWI = NDWI ./ double( samp_images(:,:,Coastline) + samp_images(:,:,NIR2) );
        NDWI_area = NDWI > NDWI_thd;
        if( sum(NDWI_area(:)) > AreaS * NDWI_area_thd),
            continue;
        end
        block_id = block_image(samp_x, samp_y);
    end
    %the name of the sampling image : blockid_x_y
    if(Image_Type ==2)
        samp_images = large_image_c( sidxs_min(iid) : sidxs_max(iid), ...
        sidys_min(iid) : sidys_max(iid), :);
    end
    samp_name = sprintf('%04d_%04d_%04d', block_id, samp_x, samp_y);
    samp_image_name = fullfile(out_image_path, samp_name);
    enviwrite(samp_images(:,:,Band_Reorder), SSize, SSize, dim(3),  samp_image_name);
    if (flag_jpg),
        jpgimage = double(samp_images(:,:,RGB_Band));
        if(Image_Type ==1),
            jpgimage(:,:,1) =  jpgimage(:,:,1) * RGB_Scale(1)  ;
            jpgimage(:,:,2) =  jpgimage(:,:,2) * RGB_Scale(2)  ;
            jpgimage(:,:,3) =  jpgimage(:,:,3) * RGB_Scale(3)  ;
        end
        jpgimage = uint8(permute(jpgimage, [2,1,3]));
        imwrite(jpgimage, [samp_image_name, '.jpg'], 'jpg');
    end
    if(mod(iid, 50) == 0)
        fprintf('Processing %.1f%%\n', iid * 100 / num_ids);
    end
end

% end
