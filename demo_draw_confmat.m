figure(1);

% % 4 USGS-DOM
% mat = rand(6);
% mat = [ 
%  92      10       0       4;
%  0      67       0       0;
%  0       1      50       0;
%  1       5       0      85
% ];

% 12 Google
% mat = [                      
% 100       0       0       2       0       0       0       0       1       0       0       0  ;
%   0      93       1       0       1       0       2       0       2       6       6       0  ;
%   0       0      83       0       2       0       0       0       0       0       3       0  ;
%   0       0       1      93       2       3       4       2       0       0       1       0  ;
%   0       1       1       0      93       0       2       0       0       0       0       0  ;
%   0       0       0       2       0      92       1      10       2       0       2       0  ;
%   0       1       0       0       0       0      86       0       1       1       0       0  ;
%   0       1       0       1       0       2       1      81       1       1       1       0  ;
%   0       0       2       1       0       3       1       7      93       0       2       0  ;
%   0       4       1       0       0       0       1       0       0      92       0       0  ;
%   0       0      10       1       2       0       2       0       0       0      84       0  ;
%   0       0       1       0       0       0       0       0       0       0       1     100  
% ];                           
% 
% % 21 ucm
% mat = [                      
% 20       0       0       0       0       0       0       0       0       0       0       0       0       0       0       0       0       0       0       0       0   ; 
%  0      18       0       0       1       0       0       0       0       0       0       0       0       0       0       0       0       0       1       1       0   ;
%  0       0      19       0       0       0       0       0       0       1       0       0       0       0       0       0       0       0       0       0       0   ;
%  0       0       0      20       0       0       0       0       0       0       0       0       0       0       0       0       0       0       0       0       0   ;
%  0       0       0       0      17       0       0       0       0       0       0       3       0       0       1       0       0       0       0       2       3   ;
%  0       0       0       0       0      20       0       0       0       0       0       0       0       0       0       0       0       0       0       0       0   ;
%  0       0       0       0       0       0      17       0       0       0       0       0       1       0       0       0       0       0       0       0       4   ;
%  0       0       0       0       0       0       0      20       0       0       0       0       0       0       0       0       1       0       1       0       0   ;
%  0       0       0       0       1       0       0       0      19       0       0       0       0       0       0       0       0       0       0       0       1   ;
%  0       0       0       0       0       0       0       0       0      19       0       0       0       0       0       0       0       0       0       0       0   ;
%  0       0       0       0       0       0       0       0       0       0      20       0       0       0       0       0       0       0       0       0       0   ;
%  0       0       0       0       0       0       0       0       0       0       0      15       0       0       0       0       0       0       0       0       0   ;
%  0       0       0       0       0       0       1       0       0       0       0       0      19       0       0       0       0       0       1       0       0   ;
%  0       0       0       0       1       0       1       0       0       0       0       1       0      20       0       0       0       0       0       0       0   ;
%  0       0       0       0       0       0       0       0       1       0       0       0       0       0      19       0       0       0       0       0       0   ;
%  0       0       0       0       0       0       0       0       0       0       0       0       0       0       0      20       0       0       0       0       0   ;
%  0       0       0       0       0       0       0       0       0       0       0       0       0       0       0       0      19       0       0       0       0   ;
%  0       2       0       0       0       0       0       0       0       0       0       0       0       0       0       0       0      20       0       0       0   ;
%  0       0       1       0       0       0       0       0       0       0       0       0       0       0       0       0       0       0      17       1       0   ;
%  0       0       0       0       0       0       1       0       0       0       0       0       0       0       0       0       0       0       0      16       1   ;
%  0       0       0       0       0       0       0       0       0       0       0       1       0       0       0       0       0       0       0       0      11   
% ];

% % 8 wuhan_iko
% mat = [
% 5       0       0       0       0       0       0       0	;
% 0       5       0       1       0       0       0       0	;
% 1       0       6       0       0       1       0       0	;
% 0       0       0       4       0       0       0       0	;
% 0       0       0       0       6       0       0       0	;
% 0       0       0       1       0       5       0       0	;
% 0       1       0       0       0       0       6       0	;
% 0       0       0       0       0       0       0       6
% ];

% % for transfer learning DCNN and 21 classes UCM dataset.
% rec_res = load('../Scene_HSRI/UCM/21Class_bmp_4tran_ft_full_227crop_acc_r1.mat');

% for coupled DCNN and 11 classes HK-WV3 dataset
%% hk
% % rec_res = load('../Scene_HSRI/HK-WV3/Testing_Samples_cnnfeat_acc_ver1.mat'); % transfer DCNN
% % rec_res = load('../Scene_HSRI/HK-WV3/Testing_Samples_cnnfeat_acc_ver3.mat'); % small DCNN
% rec_res = load('../Scene_HSRI/HK-WV3/Testing_Samples_cnnfeat_acc_ver4.mat'); % STDCNN
%% shatin
rec_res = load('../Scene_HSRI/HK-WV3-Shatin/Testing_Samples_cnnfeat_acc_ver1.mat'); % transfer DCNN
rec_res = load('../Scene_HSRI/HK-WV3-Shatin/Testing_Samples_cnnfeat_acc_ver3.mat'); % small DCNN
rec_res = load('../Scene_HSRI/HK-WV3-Shatin/Testing_Samples_cnnfeat_acc_ver4.mat'); % STDCNN

mat = rec_res.errormatrix{1}.ConfusionMatrix;
mat = mat';
bload = 1;



summat = repmat(sum(mat, 1),size(mat,1),1) ;
mat = mat ./ summat;
mat = roundn(mat, -3);            

mat_size = size(mat, 1);
% 4 USGS-DOM
if(mat_size == 4)
    clsname = {'Residential','Farm','Forest', 'Parking lot'};
end
% 12 Google
if(mat_size == 12)
    clsname = { 'agriculture', 'commercial', 'harbor',  'idle land',  'industrial', 'meadow',  'overpass', ....
                   'park', 'pond', 'residential', 'river', 'water'};
end
% 21 ucm
if(mat_size == 21)
    clsname = {'agricultural', 'airplane', 'baseball diamond', 'beach', 'buildings', ...
        'chaparral', 'dense residential', 'forest', 'freeway', 'golf course', 'harbor',...
        'intersection', 'medium residential', 'mobile home park', 'overpass', 'parking lot',...
        'river', 'runway', 'sparse residential', 'storage tanks', 'tennis courts'};
end
% 8 wuhan_iko
if(mat_size == 8)
    clsname = {'dense_residential', 'idle', 'industrial', 'medium_residential', 'parking_lot', 'commercial', ...
        'vegetation', 'water'}; 
end

% 11 HK-WV3
if(mat_size == 11)
    clsname = {'Commercial', 'Institutional', 'Port', 'D_Resid', 'S_Resid', 'Woodland', ...
        'Water', 'Open space','Vacant', 'Industrial','C_Termi'}; 
end

draw_confmat(mat',clsname);
