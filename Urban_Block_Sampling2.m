function [sx, sy, skel] = Urban_Block_Sampling2(x, y, SSize, SSpacing)
%% Urban_Block_Sampling is to generate the samples from the street block based on the skeleton extraction.
% input:
%   x, y :  the x and y set of the pixels in the current block; 
%   SSize:  the size of the samplings;
%   SSpacing: the spacing the samplings;
% output:
%   sx, sy : the x and y set of the sampling image center for the current block;
%   skel :  the skeleton of the region.
%% Write by Bei Zhao, zhaoys@cuhk.edu.hk. Jun. 21, 2016.

span_reorder_thd = SSpacing * 2 + SSize;

semiSSize = floor(SSize / 2);

xmin = min(x); ymin = min(y);
xmax = max(x); ymax = max(y);
xorign = xmin; yorign = ymin;
x0 = x - xorign+1;
y0 = y - yorign+1;
sxmaxnum = floor(( xmax - xmin + 1) / SSpacing) + 1;
symaxnum = floor(( ymax - ymin + 1) / SSpacing) + 1;
maxnum = sxmaxnum * symaxnum;
sx_cand = zeros(maxnum, 1);
sy_cand = zeros(maxnum, 1);

xspan = xmax-xmin + 1;
yspan = ymax-ymin + 1;
img_flag = false(xspan, yspan);
ids0 = x0  + (y0-1) * xspan;
img_flag(ids0) = true;
% img_flag = imread('circles.png');
ske = bwmorph(img_flag,'skel',Inf);
skel = ske;

count = 1;
while(true),
    [xsk, ysk] = find(ske);
    num_skp = numel(xsk);
    if num_skp == 0
        break;
    else
        fprintf('Number of points = %d\n', num_skp);
    end
    if(xspan < span_reorder_thd && yspan < span_reorder_thd )
        [xsk, ysk] = reorder(xsk, ysk);
    end
    for iskp = 1 : num_skp,
        if(img_flag( xsk(iskp), ysk(iskp) ))
            sx_cand(count) = xsk(iskp) + xorign -1;
            sy_cand(count) = ysk(iskp) + yorign -1;
            count = count  + 1;
            [img_flag, ske] = suppress(img_flag, ske, xsk(iskp), ysk(iskp), semiSSize, SSize);
        else
            ske( xsk(iskp), ysk(iskp)) = false;
        end
    end
end

count = count - 1 ;
sx = sx_cand(1:count);
sy = sy_cand(1:count);

end

function [out_flag, out_ske] = suppress(in_flag, in_ske, xcen, ycen, semiSSize, SSize)
    dim = size(in_flag);
    xcen_min = xcen - semiSSize + 1;
    xcen_max = xcen_min + SSize - 1 ;
    xcen_min = max(xcen_min, 1);
    xcen_max = min(xcen_max, dim(1));
    
    ycen_min = ycen - semiSSize + 1;
    ycen_max = ycen_min + SSize - 1 ;
    ycen_min = max(ycen_min, 1);
    ycen_max = min(ycen_max, dim(2));
   
    out_flag =  in_flag;
    out_ske = in_ske;
   
    out_flag(xcen_min:xcen_max, ycen_min:ycen_max) = false;
    out_ske(xcen_min:xcen_max, ycen_min:ycen_max) = false;
    
    if(xcen_min > 1),
        out_ske(xcen_min -1 , ycen_min:ycen_max) = out_flag(xcen_min -1, ycen_min:ycen_max) > 0;
    end
    if(xcen_max+1 <=  dim(1)),
        out_ske(xcen_max+1, ycen_min:ycen_max) = out_flag(xcen_max+1, ycen_min:ycen_max) > 0;
    end
    if(ycen_min > 1),
        out_ske(xcen_min:xcen_max, ycen_min-1) = out_flag(xcen_min:xcen_max, ycen_min-1) > 0;
    end
    if(ycen_max+1 <=  dim(2)),
        out_ske(xcen_min:xcen_max, ycen_max+1) = out_flag(xcen_min:xcen_max, ycen_max+1) > 0;
    end
end

function [xout, yout] = reorder(xin, yin)
    xcen = floor(mean(xin));
    ycen = floor(mean(yin));
    dist_cen = (xin - xcen).^2 + (yin - ycen).^2 ;
    [~, index] = sort(dist_cen);
    xout = xin(index);
    yout = yin(index);
end
