clc; clear all;
%% demo_transfer_fe
% Demonstration CNN-FE/MLP for land-use classification
%% bei zhao, zhaoys@cuhk.edu.hk, March 21, 2016
argu_type = 0; % 0, no arugmentation; 1, corner argumentation (5 times); 2 corner and flip (10 times)
dataset = 2; % 1: 21 UCM; 2 : 8 Wuhan-IKONOS;

token = '*.bmp';
crop_shape = [227, 227];

rand_num = 5;
tr_samples_per = 0.8;
test_interval = 100;
iternum = 10000;

mean_file = 'lu-model/ilsvrc_2012_mean.mat';
weights_file = 'lu-model/bvlc_alexnet/bvlc_alexnet.caffemodel';
if (dataset == 1)
    lu_filepath = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/UCM/21Class_bmp'; % File path of the dataset
    tr_txt = 'lu-model/transfer_fe/21Class_4train_ind.txt';
    val_txt = 'lu-model/transfer_fe/21Class_4val_ind.txt';
    rand_sample_file = 'lu-model/rand_sample.mat';

    alex_model_proto = 'lu-model/transfer_fe/4feat_ext_cnn.prototxt';
    model_tran_proto = 'lu-model/transfer_fe/4feat_ext_tran.prototxt';
    solver_proto = 'lu-model/transfer_fe/4solver_transfer_fixed.prototxt';
    weights_tran_file = ['lu-model/transfer_fe/caffe_alexnet_fe_4train_iter_', num2str(iternum), '.caffemodel'];
else
    lu_filepath = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/Wuhan-IKONOS/ImageSamplesPCA'; 
    tr_txt =  'wuhan/transfer_fe/Wuhan_4train_ind.txt';
    val_txt =  'wuhan/transfer_fe/Wuhan_4val_ind.txt';
    rand_sample_file = 'wuhan/rand_sample.mat';

    alex_model_proto = 'wuhan/transfer_fe/4feat_ext_cnn.prototxt';
    model_tran_proto = 'wuhan/transfer_fe/4feat_ext_tran.prototxt';
    solver_proto = 'wuhan/transfer_fe/4solver_transfer_fixed.prototxt';
    weights_tran_file = ['wuhan/transfer_fe/caffe_alexnet_fe_4train_iter_', num2str(iternum), '.caffemodel'];
end

gpu_id  = 0;
layer_rec = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7-ft', 'fc8-ft','fc9-ft'};
layer_rec_num = numel(layer_rec);
%% Prepare data;
if (argu_type == 0),
	num_mul = 1;
elseif(argu_type == 1) 
	num_mul = 5; 
else % argu_type ==  2
    num_mul = 10;
end

if exist('caffe-master/matlab/+caffe', 'dir')
  addpath('caffe-master/matlab');
else
  error('Please run this demo from deep_learning/LU');
end

meandload = load(mean_file);
mean_data = meandload.mean_data;
addpath('feature_ext');
% DataInfo = ImageSet2Mat(lu_filepath, token,  crop_shape, argu_type, mean_data);
load([lu_filepath,'_', num2str(crop_shape(1)), 'crop_',  num2str(num_mul), 'times_mat_info.mat']);
rmpath('feature_ext');

class_imgnum = DataInfo.class_imgnum;
total_imgnum = sum(class_imgnum) * num_mul;
class_num = DataInfo.class_num;

%% feature extraction and transfer learning
disp('load images');
tic;
load( DataInfo.mat_path{1});
cur_imgnum = class_imgnum(1)*num_mul;
image_channels = size(CropArgData, 3);
imagedata = zeros(crop_shape(1), crop_shape(2), image_channels, total_imgnum, 'single');
imagedata(:,:,:,1:cur_imgnum) = CropArgData(:,:,:,:);
imid = cur_imgnum+1;
for is = 2 : class_num,
    load( DataInfo.mat_path{is});
    cur_imgnum = class_imgnum(is)*num_mul;
    imagedata(:,:,:,imid:imid+cur_imgnum-1) = CropArgData(:,:,:,:);
    imid = imid + cur_imgnum;
end
clear CropArgData;
times1 = toc;

tic;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(alex_model_proto, weights_file, 'test'); % create net and load weights

cnn_feat = zeros(1, 1, 4096, total_imgnum, 'single');
disp('Extract cnn features');
for im = 1 : total_imgnum,
    data = imagedata(:,:,:,im);
    res = net.forward({data});
    cnn_feat(:,:,:,im) = res{1}; % net.blobs('fc6').get_data();
end
caffe.reset_all();
times2 = toc;

clear net res;
save([lu_filepath,'_', num2str(crop_shape(1)), 'crop_',  num2str(num_mul), 'times_cnnfeat.mat'], 'cnn_feat');

%% transfer learning
tic;
errormatrix = cell(rand_num,1);
oa_acc = zeros(rand_num,1);

% sample_perm_all = zeros(rand_num, class_imgnum(1));
% for randi = 1 : rand_num,
%    sample_perm_all(randi,:) = randperm(class_imgnum(1));
% end
% save(rand_sample_file, 'sample_perm_all');
load(rand_sample_file);
for randi = 1 : rand_num,
    sample_perm = sample_perm_all(randi,:);
    
    train_ids = zeros(class_imgnum(1), class_num);
    train_ids( sample_perm( 1 : floor( class_imgnum(1)* tr_samples_per ) ),  :) = 1;
    train_ids = repmat( train_ids(:), [1, num_mul] );
    train_ids = train_ids';
    train_ids = logical(train_ids(:));
    num_train = sum(train_ids);
    num_train_pc = class_imgnum(1)* tr_samples_per;
    num_test = total_imgnum - num_train;
    num_test_pc = class_imgnum(1) - num_train_pc;
    
    sample_label = DataInfo.mat_label;
    tr_cnn_feat = cnn_feat(:,:,:,train_ids);
    tr_cnn_feat_label = sample_label(train_ids) - 1; % the start label is from 0;
    order = 1:num_train;
    order = reshape(order, [num_mul, num_train_pc, class_num]);
    order = permute(order, [3, 2, 1]);
    tr_cnn_feat = tr_cnn_feat(:,:,:,order(:));
    tr_cnn_feat_label = tr_cnn_feat_label(order(:));
    
    te_cnn_feat = cnn_feat(:,:,:,~train_ids);
    te_cnn_feat_label = sample_label(~train_ids) - 1;
    order = 1:num_test;
    order = reshape(order, [num_mul, num_test_pc, class_num]);
    order = permute(order, [3,2,1]);
    te_cnn_feat = te_cnn_feat(:,:,:,order(:));
    te_cnn_feat_label = te_cnn_feat_label(order(:));
    
%% 1. cnn-fe/mlp
    % training
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
    solver = caffe.Solver(solver_proto);
    net = solver.net;
    net.layers('data').set_memdata(tr_cnn_feat, tr_cnn_feat_label, num_train);
    clear tr_cnn_feat;
    test_net = solver.test_nets(1);
    test_net.layers('data').set_memdata(te_cnn_feat , te_cnn_feat_label, num_test);
    
    record_num = floor(iternum/test_interval);
    rec_loss = zeros(record_num,2); % training loss and test loss
    for ir = 1 : record_num,
        solver.step(test_interval);
        rec_loss(ir, 1) = solver.net.blobs('accuracy').get_data();
        rec_loss(ir, 2) = solver.test_nets(1).blobs('accuracy').get_data();
    end
    caffe.reset_all();
    % test
    caffe.set_device(gpu_id);
    net = caffe.Net(model_tran_proto, weights_tran_file, 'test'); % create net and load weights
    test_pro = zeros(class_num, num_test, 'single');
    disp('Extract transfer features');
    for im = 1 : num_mul: num_test,        
        data = te_cnn_feat(:,:,:,im: im+num_mul-1) ;
        net.forward({data});
        test_pro(:,im:im+num_mul-1) =  net.blobs('prob').get_data();
    end
    caffe.reset_all();
    num_test_images = num_test / num_mul;
    test_pro = reshape(test_pro, [class_num, num_test_images, num_mul]);
    test_pro = mean(test_pro, 3);
    [~, pred_labels] = max(test_pro);

%% 2. cnn-fe/svm
%     addpath('liblinear/matlab');
% %     svmparam = train(tr_cnn_feat_label(:), sparse(reshape(double(tr_cnn_feat),[4096, num_train])), '-C -q', 'col');
% %     svmmodel = train(tr_cnn_feat_label(:), sparse(reshape(double(tr_cnn_feat),[4096, num_train])), ['-c ', num2str(svmparam(1)), ' -q'], 'col');
%     svmmodel = train(tr_cnn_feat_label(:), sparse(reshape(double(tr_cnn_feat),[4096, num_train])), '-c 9 -q', 'col');
%     [pred_labels, accuracy, ~] = predict(te_cnn_feat_label, sparse(reshape(double(te_cnn_feat),[4096, num_test])), svmmodel, '-q', 'col');
%     pred_labels = pred_labels + 1;
%     rmpath('liblinear/matlab');

%% get the confusion matrix and the accuracy
    pred_labels = pred_labels(:);
    acc = assessment(int32(te_cnn_feat_label(1:num_test_images) + 1), int32(pred_labels),'class');
    fprintf('Round %d:  OA = %.2f%%\n', randi, acc.OA);
    errormatrix{randi} = acc;
    oa_acc(randi) = acc.OA;
    clear data net;
end
times3 = toc;
fprintf('Total cost times: %.1f (s)\n', times1+ times2 + times3);

% save([lu_filepath, '_4tran_fe_svm_' , num2str(crop_shape(1)), 'crop_',  num2str(num_mul), 'times_acc.mat'],...
%     'oa_acc', 'errormatrix', 'test_pro', 'rec_loss');
save([lu_filepath, '_4tran_fe_mlp_' , num2str(crop_shape(1)), 'crop_',  num2str(num_mul), 'times_acc.mat'],...
    'oa_acc', 'errormatrix', 'test_pro', 'rec_loss');

fprintf('OA: mean = %.2f, std = %.2f (%%)\n', mean(oa_acc), std(oa_acc));

rmpath('caffe-master/matlab');
