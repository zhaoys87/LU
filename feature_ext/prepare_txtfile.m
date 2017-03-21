function prepare_txtfile(feaSet, tr_ids, val_ids, tr_txt, val_txt)
%% prepare_txtfile(feaSet, tr_ids, val_ids, txtfilename)
%       write a txtfile to input into the imagedata layer of caffe.
% input:
%   feaSet : the dataset info of images (using SearchFolder2Big.m );
%   tr_ids : the training image indices; classnum cells;
%   val_ids : the validation image indices; classnum cells;
%   tr_txt : the training txt file;
%   val_txt : the validation txt file;
%% zhaoys@cuhk.edu.hk, Apr. 11, 2016

class_num   = feaSet.class_num ;       % number of classes
class_path  = feaSet.class_path;       % the path of each class
img_name    = feaSet.img_name ;        % contain the pathes for each image of each class
class_imgnum =  feaSet.class_imgnum  ;   % number of images contained in each class

fid1 = fopen( tr_txt, 'w');
fid2 = fopen( val_txt, 'w');
count = 0; 
for ic = 1 : class_num;
    cur_tr_ids = tr_ids{ic}; 
    cur_val_ids = val_ids{ic};
    num_cur_trids = numel(cur_tr_ids);
    num_cur_valids = numel(cur_val_ids);
    for iid = 1 : num_cur_trids,
        filename = fullfile(class_path{ic}, img_name{ count + cur_tr_ids(iid) });
        fprintf(fid1, '%s %d\n', filename, ic -1);
    end
    for iid = 1 : num_cur_valids,
        filename = fullfile(class_path{ic}, img_name{ count + cur_val_ids(iid) });
        fprintf(fid2, '%s %d\n', filename, ic -1);
    end
    count = count + class_imgnum(ic);
end

fclose(fid1);
fclose(fid2);
