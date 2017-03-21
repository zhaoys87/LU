function CropArgData = ImageCropArgu(imagedata, crop_shape, argu_type, image_mean)
%% function CropArgData = ImageCropArgu(imagedata, crop_shape, argu_type, image_mean)
% 	Crop image with crop_shape, and do the argument with the argu_type
%input parameter:
%   imagedata :  [width, height, bands]
%   crop_shape  :  [width, height]
%   image_mean :  the image mean (w * h * b)
%   argu_type : the type of argument, 0, no argument (default); 1, corner argument
%       (5 times); 2, argumentation with corner sampling and
%       horizonal flip (10 times).
%output parameter:
%   CropArgData: the cropped and argumented data [width, height, band, mult ]
%% Writed by Bei Zhao on Mar. 23, 2016.

if(~exist('argu_type','var'))
    argu_type = 0;
end
if(~exist('crop_shape','var'))
    crop_shape = [224, 224];
end

if(~exist('image_mean','var'))
    bexist_mean = 0;
    imh = 256;
    imw = imh;
else
    bexist_mean = 1;
    [imh, imw, ~] = size(image_mean);
end

%% read files
if (argu_type == 0),
	num_mul = 1;
elseif(argu_type == 1) 
	num_mul = 5; 
else % argu_type ==  2
    num_mul = 10;
end

[cur_imh, cur_imw, cur_imb] = size(imagedata);
if(cur_imh ~= imh ||  cur_imw ~= imw)
    imagedata = single( imresize(imagedata, [imh, imw]) ) ;
    [cur_imh, cur_imw, cur_imb] = size(imagedata);
end

if(bexist_mean)
    imagedata = single(imagedata) - image_mean;
end

CropArgData = zeros( crop_shape(1), crop_shape(2), cur_imb, num_mul, 'single' );

if (num_mul == 1)
    CropArgData(:,:, :, 1) = imresize(imagedata, crop_shape);
else % num_mul == 5, 10
    % center
    CenW = floor(cur_imw/2 - crop_shape(1)/2 + 0.5) ;
    CenH = floor(cur_imh/2 - crop_shape(2)/2 + 0.5) ;
    CropArgData(:,:,:, 1) = imagedata(CenW:CenW+crop_shape(1)-1, CenH:CenH+crop_shape(2)-1, :);
	% up-left
	CropArgData(:,:,:, 2) = imagedata(1:crop_shape(1), 1:crop_shape(2), :);
	% up-right
	StartW = cur_imw - crop_shape(1) + 1 ;
	CropArgData(:,:,:, 3) = imagedata(StartW:end, 1:crop_shape(2), :);
	% down-left
	StartH = cur_imh - crop_shape(2) + 1 ;
	CropArgData(:,:,:, 4) = imagedata(1:crop_shape(1), StartH:end, :);
	% down-right
	StartW = cur_imw - crop_shape(1) + 1 ;
	StartH = cur_imh - crop_shape(2) + 1 ;
	CropArgData(:,:,:, 5) = imagedata(StartW:end, StartH:end, :);
    
    if (num_mul == 10)
        for im = 1 : 5,
            CropArgData(:,:,:, im+5) = CropArgData(end:-1:1, :, :, im);
        end
    end

end





