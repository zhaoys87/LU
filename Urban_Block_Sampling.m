function [sx, sy] = Urban_Block_Sampling(x, y, SSize, SSpacing)
%% Urban_Block_Sampling is to generate the samples from the street block.
% input:
%   x, y :  the x and y set of the pixels in the current block; 
%   SSize:  the size of the samplings;
%   SSpacing: the spacing the samplings;
% output:
%   sx, sy : the x and y set of the sampling image center for the current block; 
%% Write by Bei Zhao, zhaoys@cuhk.edu.hk. Jun. 7, 2016.
xmin = min(x); ymin = min(y);
xmax = max(x); ymax = max(y);
sxmaxnum = floor(( xmax - xmin + 1) / SSpacing) + 1;
symaxnum = floor(( ymax - ymin + 1) / SSpacing) + 1;
maxnum = sxmaxnum * symaxnum;
sx_cand = zeros(maxnum, 1);
sy_cand = zeros(maxnum, 1);
count = 1;

% The center sampling image
semiSSize = floor(SSize / 2);
xcen0 = floor(mean(x));

% Iteration right lines;
xcen = xcen0;  
while( xcen <= xmax) % for each lines
    % center
    [ycen, yids]= find_center(xcen, x, y, semiSSize, SSize);
    if(isempty(ycen)),
        xcen = xcen + semiSSize;
        continue;
    end
    if( in_region(x(yids), y(yids), xcen, ycen) )
        sx_cand(count) = xcen;
        sy_cand(count) = ycen;
        count = count  + 1;
    end    
    % up
    ymin_cur = min( y(yids) );
    ycur = ycen - SSpacing;
    while(ycur >= ymin_cur)
        if( in_region(x(yids), y(yids), xcen, ycur) )
            sx_cand(count) = xcen;
            sy_cand(count) = ycur;
            count = count  + 1;
        end
        ycur = ycur - SSpacing;
    end
    % down
    ymax_cur = max( y(yids) );
    ycur = ycen + SSpacing;
    while(ycur <= ymax_cur)
        if( in_region(x(yids), y(yids), xcen, ycur) )
            sx_cand(count) = xcen;
            sy_cand(count) = ycur;
            count = count  + 1;
        end
        ycur = ycur + SSpacing;
    end
    %update the center lines
    xcen = xcen + semiSSize;
end

% Iteration left lines;
xcen = xcen0 - semiSSize;  
while( xcen >= xmin) % for each lines
    % center
    [ycen, yids]= find_center(xcen, x, y, semiSSize, SSize);
    if(isempty(ycen)),
        xcen = xcen - semiSSize;
        continue;
    end
    if( in_region(x(yids), y(yids), xcen, ycen) )
        sx_cand(count) = xcen;
        sy_cand(count) = ycen;
        count = count  + 1;
    end    
    % up
    ymin_cur = min( y(yids) );
    ycur = ycen - SSpacing;
    while(ycur >= ymin_cur)
        if( in_region(x(yids), y(yids), xcen, ycur) )
            sx_cand(count) = xcen;
            sy_cand(count) = ycur;
            count = count  + 1;
        end
        ycur = ycur - SSpacing;
    end
    % down
    ymax_cur = max( y(yids) );
    ycur = ycen + SSpacing;
    while(ycur <= ymax_cur)
        if( in_region(x(yids), y(yids), xcen, ycur) )
            sx_cand(count) = xcen;
            sy_cand(count) = ycur;
            count = count  + 1;
        end
        ycur = ycur + SSpacing;
    end
    %update the center lines
    xcen = xcen - semiSSize;
end
count = count - 1 ;
sx = sx_cand(1:count);
sy = sy_cand(1:count);
end

function [ ycen, yids ] = find_center(xcen, x, y, semiSSize, SSize)
    xcen_min = xcen - semiSSize;
    xcen_max = xcen_min + SSize - 1 ;
    yids = x >= xcen_min & x <= xcen_max;
    if (isempty( y(yids) )),
        ycen = [];
    else
        ycen = floor(mean( y(yids) ));
    end
end

function flag = in_region(x, y, sx, sy)
    ysels = y(x == sx);
    flag = sum(ysels == sy) > 0;
end