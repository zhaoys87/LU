function feaSet = SearchFolder2Big( directory, token )
%SearchFolder : search files in the subfolder of directory
% Input parameter
%   directory : the directory to be search in
%   token : the file contain the token. such as '*.tif'
% Output parameters
%   feaSet;
%     feaSet.class_num        % number of classes
%     feaSet.class_name       % name of each class
%     feaSet.class_path       % the path of each class
%     feaSet.img_name         % contain the pathes for each image of each class
%     feaSet.class_imgnum     % number of images contained in each class
%     feaSet.img_label        % label of each image contained in img_name
% Examples
%   feaSet = SearchFolder2('E:\dataset\google', '*.tif');
    if ~isdir(directory) 
        error('the directory should be a file directory!');
    end
    sub = dir(directory);
    sub_num = size(sub,1)-2; % ~strcmp(subname, '.') & ~strcmp(subname, '..')
  
    feaSet =  struct;
    feaSet.class_num = sub_num;
    feaSet.class_name = cell(sub_num,1);  % name of each class
    feaSet.class_path = cell(sub_num,1);  % name of each class
    feaSet.img_name = cell(1);            % contain the pathes for each image of each class
    feaSet.class_imgnum = [];             % number of images contained in each class
    feaSet.img_label = [];               % label of each image contained in img_name
    file_id = 0;
    for i=1 : sub_num
        feaSet.class_name{i} = sub(i+2).name;
        feaSet.class_path{i} = fullfile(directory,  feaSet.class_name{i});
        
        folderfile = dir( fullfile(feaSet.class_path{i}, token)); % search files in the folder
        
        file_num_per_c = size(folderfile, 1);
        feaSet.class_imgnum(i) = file_num_per_c;
        for jj = 1: file_num_per_c
            feaSet.img_name{file_id+1} = folderfile(jj).name;
            feaSet.img_label(file_id+1) = i;
            file_id = file_id + 1;
        end

    end
    
end

