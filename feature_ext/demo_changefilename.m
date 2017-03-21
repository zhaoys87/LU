lu_filepath = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/UCM/21Class'; % File path of the dataset
%% change file name;
% token = '*.TIFF';
% changedtoken = '.tif';
% 
% feaSet = SearchFolder2Big( lu_filepath, token );
% class_path = feaSet.class_path;
% img_name = feaSet.img_name;
% class_imgnum = feaSet.class_imgnum;
% total_imgnum = sum(class_imgnum);
% class_num = feaSet.class_num;
% 
% count = 0;
% for ic = 1 : class_num,
%     for iid = 1 : class_imgnum(ic),
%         filename = fullfile(class_path{ic}, img_name{ count + iid });
%         [fp,fn,suf] = fileparts(filename);
%         newfilename = fullfile(fp,[fn, changedtoken]);
%         movefile(filename, newfilename);
%     end
%     count= count + class_imgnum(ic);
% end

%% convert format
token = '*.tif';
changedtoken = 'bmp';
outpath = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/UCM/21Class_bmp';

feaSet = SearchFolder2Big( lu_filepath, token );
class_path = feaSet.class_path;
img_name = feaSet.img_name;
class_imgnum = feaSet.class_imgnum;
total_imgnum = sum(class_imgnum);
class_num = feaSet.class_num;

count = 0;
for ic = 1 : class_num;
    for iid = 1 : class_imgnum(ic),
        filename = fullfile(class_path{ic}, img_name{ count + iid });
        [fp,fn,suf] = fileparts(filename);
        [fp,fpsub] = fileparts(fp);
        newfilepath = fullfile(outpath,fpsub);
        newfilename = fullfile(newfilepath, [fn, '.', changedtoken]);
        imagedata = imread(filename);
        if( ~exist(newfilepath) )
            mkdir(newfilepath);
        end
        imwrite(imagedata, newfilename, changedtoken);
    end
    count= count + class_imgnum(ic);
end
