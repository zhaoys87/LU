
dataset = 2; 
colormapfile = 'shatin-wv3/transfer_ft/results/colormap.mat';
if dataset == 2,
    pro_image_path = 'shatin-wv3/transfer_ft/results/shatin-cls-ver4_samp2_pro_rev';
else
    pro_image_path = 'shatin-wv3/transfer_ft/results/hk-wv3-cls-ver4_samp2_pro_rev';
end
%% subplot
% [pro_image, dim] = freadenvi(pro_image_path);
% pro_image = reshape( single(pro_image), dim);
% scrsz = get(0,'ScreenSize');
% figure('Position',[1, 1, scrsz(3)*4/5 scrsz(4)*4/5])
% clf();
% hold on;
% % Number of rows and columns of axes
% ncols = 4;
% nrows = 3;
% subsize = 0.85;
% hspacing = 0.04;
% vspacing = 0.04;
% % w and h of each axis in normalized units
% axisw = (1 / ncols) * subsize;
% axish = (1 / nrows) * subsize;
% 
% for imid = 1 : dim(3),
%     % calculate the row and column of the subplot
%     row = floor( (imid-1) / ncols ) + 1;
%     col = mod( imid-1, ncols ) + 1;
%     % calculate the left, bottom coordinate of this subplot
%     axisl = (axisw+hspacing) * (col-1);
%     axisb = (axish+vspacing) * (nrows-row) + vspacing/2;
%     %  plot the subplot
%     subplot('position', [axisl, axisb, axisw, axish] );
%     % display image
%     curimg = pro_image(:,:,imid);
%     imshow(double(curimg'));
%     title(clsname{imid},'FontSize',16);
%     colorbar('FontSize',14);                                                  
% end
%                                                                       
% colormap(jet(128));

%% write classification file; generate the class images
[pro_image, dim] = freadenvi(pro_image_path);
pro_image = reshape( single(pro_image), dim);
colormat = load(colormapfile);
num_class = size(colormat.colormap, 1);
span_value = 1/num_class;
denseslice = zeros(dim(1), dim(2), dim(3), 'uint8');
marker = true(dim(1), dim(2), dim(3));
disp('generate the class images');
tic;
for ic = 0:num_class-1,
    value1 = span_value * ic;
    value2 = span_value * (ic+1);
    ids = (pro_image(marker) > value1)  & (pro_image(marker) <= value2) ;
    cursl = denseslice(marker);
    cursl(ids) = ic;
    denseslice(marker) = cursl;
    cursl = marker(marker);
    cursl(ids) = false;
    marker(marker) = cursl;
    fprintf('processing %.1f\n', (ic+1)*100/num_class);
end
toc;

% write images
disp('write images');
tic;
lookup = mat2str(colormat.colormap);
lookup = strrep(lookup, ' ', ', ');
lookup = strrep(lookup, ';', ', ');
for ib = 1 : dim(3),
    fname = [pro_image_path num2str(ib)];
    wfid = fopen(fname,'w');
    if wfid == -1
        i=-1;
    end
    fwrite(wfid,denseslice(:,:,ib),'uint8');
    fclose(wfid);
    % Write header file
    fid = fopen(strcat(fname,'.hdr'),'w');
    if fid == -1
       i=-1;
    end
    fprintf(fid,'%s \n','ENVI');
    fprintf(fid,'%s \n','description = {');
    fprintf(fid,'%s \n','Exported from MATLAB}');
    fprintf(fid,'%s %i \n','samples =',dim(1));
    fprintf(fid,'%s %i \n','lines   =',dim(2));
    fprintf(fid,'%s \n','bands   = 1');
    fprintf(fid,'%s \n','data type = 1');
    fprintf(fid,'%s \n','interleave = bsq');
    fprintf(fid,'%s \n','header offset = 0');
    fprintf(fid,'%s \n','file type = ENVI Classification');
    fprintf(fid,'%s %i \n','classes =', num_class);
    fprintf(fid,'%s \n','class lookup = {');
    fprintf(fid,'%s } \n',lookup);
    fclose(fid);
end
toc;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             