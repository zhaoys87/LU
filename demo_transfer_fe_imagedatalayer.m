clc; clear all;
%% demo_transfer_featureExt
% Demo of CNN-FE/MLP-R for Land-use classification
%% bei zhao, zhaoys@cuhk.edu.hk, March 21, 2016
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
    tr_txt = 'lu-model/transfer_fe/21Class_5train_ind.txt';
    val_txt = 'lu-model/transfer_fe/21Class_5val_ind.txt';
    rand_sample_file = 'lu-model/rand_sample.mat';

    solver_proto = 'lu-model/transfer_fe/5solver_transfer_fixed.prototxt';
    weights_tran_file = ['lu-model/transfer_fe/caffe_alexnet_fe_5train_iter_', num2str(iternum), '.caffemodel'];
    model_tran_proto = 'lu-model/transfer_fe/5feat_ext_tran.prototxt';
else
    lu_filepath = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/Wuhan-IKONOS/ImageSamplesPCA'; 
    tr_txt =  'wuhan/transfer_fe/Wuhan_5train_ind.txt';
    val_txt =  'wuhan/transfer_fe/Wuhan_5val_ind.txt';
    rand_sample_file = 'wuhan/rand_sample.mat';

    alex_model_proto = 'wuhan/transfer_fe/5feat_ext_cnn.prototxt';
    model_tran_proto = 'wuhan/transfer_fe/5feat_ext_tran.prototxt';
    solver_proto = 'wuhan/transfer_fe/5solver_transfer_fixed.prototxt';
    weights_tran_file = ['wuhan/transfer_fe/caffe_alexnet_fe_5train_iter_', num2str(iternum), '.caffemodel'];
end

gpu_id  = 0;
layer_rec = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7-ft', 'fc8-ft','fc9-ft'};
layer_rec_num = numel(layer_rec);

mean_load = load(mean_file);
mean_data = mean_load.mean_data;
mean_size = zeros(2,1);
[mean_size(1), mean_size(2), ~] = size(mean_data);
%% Prepare data;
if exist('caffe-master/matlab/+caffe', 'dir')
  addpath('caffe-master/matlab');
else
  error('Please run this demo from deep_learning/LU');
end
addpath('feature_ext');

feaSet = SearchFolder2Big( lu_filepath, token );
class_imgnum = feaSet.class_imgnum;
total_imgnum = sum(class_imgnum);
class_num = feaSet.class_num;
class_path  = feaSet.class_path;       % the path of each class
img_name    = feaSet.img_name ;        % contain the pathes for each image of each class

%% transfer learning
%% feature extraction and training the classifier
errormatrix = cell(rand_num,1);
oa_acc = zeros(rand_num,1);

% sample_perm_all = zeros(rand_num, class_imgnum(1));
% for randi = 1 : rand_num,
%    sample_perm_all(randi,:) = randperm(class_imgnum(1));
% end
% save(rand_sample_file, 'sample_perm_all');
load(rand_sample_file);
for randi = 1 : rand_num,
    tic;
    sample_perm = sample_perm_all(randi,:);
    
    tr_ids = cell(class_num,1);
    val_ids = cell(class_num,1);
    sel_tr_ids = sample_perm( 1 : floor( class_imgnum(1)* tr_samples_per ) );
    sel_val_ids = sample_perm( floor( class_imgnum(1)* tr_samples_per ) + 1 : end );
    for ic = 1 : class_num,
        tr_ids{ic} = sel_tr_ids;
        val_ids{ic} = sel_val_ids;
    end
    num_train = numel(sel_tr_ids)*class_num;
    num_test = total_imgnum - num_train;
    prepare_txtfile(feaSet, tr_ids, val_ids, tr_txt, val_txt);
    
    %training
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
    solver = caffe.Solver(solver_proto);
    solver.net.copy_from(weights_file);
    record_num = floor(iternum/test_interval);
    rec_loss = zeros(record_num,2); % training loss and test loss
    for ir = 1 : record_num,
        solver.step(test_interval);
        rec_loss(ir, 1) = solver.net.blobs('accuracy').get_data();
        rec_loss(ir, 2) = solver.test_nets(1).blobs('accuracy').get_data();
    end
    caffe.reset_all();
   
    % get the confusion matrix and the accuracy
    caffe.set_device(gpu_id);
    net = caffe.Net(model_tran_proto, weights_tran_file, 'test'); % create net and load weights
    test_pro = zeros(class_num, num_test, 'single');
    te_label = zeros(num_test, 1,'int32');
    count  = 0;
    counttest = 1;
    CenW = floor(mean_size(1)/2 - crop_shape(1)/2 + 0.5) ;
    CenH = floor(mean_size(2)/2 - crop_shape(2)/2 + 0.5) ;
    for ic = 1 : class_num;
        cur_val_ids = val_ids{ic};
        num_cur_valids = numel(cur_val_ids);
        for iid = 1 : num_cur_valids,
            filename = fullfile(class_path{ic}, img_name{ count + cur_val_ids(iid) });
            imageval = imread(filename);
            imageval = imresize( imageval(:,:,[3,2,1]), [mean_size(1), mean_size(2)]);
            imageval =  permute( single(imageval),[2, 1, 3] )- mean_data;
            cropdata = imageval(CenW:CenW+crop_shape(1)-1, CenH:CenH+crop_shape(2)-1, :);
            net.forward({cropdata});
            test_pro(:,counttest) =  net.blobs('prob').get_data();
            te_label(counttest) = ic;
            counttest = counttest + 1;
        end
        count = count + class_imgnum(ic);
    end
    caffe.reset_all();
    
    test_pro = reshape(test_pro, [class_num, num_test]);
    test_pro = mean(test_pro, 3);
    [~, pred_labels] = max(test_pro);
    pred_labels = pred_labels(:);
    acc = assessment( te_label(1:num_test), int32(pred_labels),'class');
    fprintf('Round %d:  OA = %.2f%%\n', randi, acc.OA);
    errormatrix{randi} = acc;
    oa_acc(randi) = acc.OA;
    clear data net;
    
    spent_times = toc;
    fprintf('The training time (s) : %.1f\n', spent_times);
end

save([lu_filepath, '_5tran_fe_' , num2str(crop_shape(1)), 'crop_acc.mat'],...
    'oa_acc', 'errormatrix', 'test_pro', 'rec_loss');
fprintf('OA: mean = %.2f, std = %.2f (%%)\n', mean(oa_acc), std(oa_acc));

% load([lu_filepath, '_5tran_fe_227crop_',  num2str(num_mul), 'times_acc.mat']);

rmpath('caffe-master/matlab');
rmpath('feature_ext');