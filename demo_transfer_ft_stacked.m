clc; clear all;
%% demo_transfer_finetuning
% Demonstration of CNN-FT-Full for land-use classification
%% bei zhao, zhaoys@cuhk.edu.hk, March 21, 2016
%% input data
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
    tr_txt =  'lu-model/transfer_finetuning/21Class_4train_ind.txt';
    val_txt =  'lu-model/transfer_finetuning/21Class_4val_ind.txt';
    rand_sample_file = 'lu-model/rand_sample.mat';

    model_tran_proto = 'lu-model/transfer_finetuning/4feat_tran.prototxt';
    solver_proto = 'lu-model/transfer_finetuning/4solver_transfer_ft.prototxt';
    weights_tran_file = ['lu-model/transfer_finetuning/caffe_alexnet_ft_4train_iter_', num2str(iternum), '.caffemodel'];

    model_alexnet_proto = 'lu-model/transfer_finetuning/11fe_feat_ext_cnn.prototxt';
    solver_tran11_proto = 'lu-model/transfer_finetuning/11fe_solver_transfer_fixed.prototxt';
    weights_tran11_file = ['lu-model/transfer_finetuning/caffe_alexnet_fe_11train_iter_', num2str(iternum), '.caffemodel'];
else
    lu_filepath = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/Wuhan-IKONOS/ImageSamplesPCA'; 
    tr_txt =  'wuhan/transfer_finetuning/Wuhan_4train_ind.txt';
    val_txt =  'wuhan/transfer_finetuning/Wuhan_4val_ind.txt';
    rand_sample_file = 'wuhan/rand_sample.mat';

    model_tran_proto = 'wuhan/transfer_finetuning/4feat_tran.prototxt';
    solver_proto = 'wuhan/transfer_finetuning/4solver_transfer_ft.prototxt';
    weights_tran_file = ['wuhan/transfer_finetuning/caffe_alexnet_ft_4train_iter_', num2str(iternum), '.caffemodel'];

    model_alexnet_proto = 'wuhan/transfer_finetuning/11fe_feat_ext_cnn.prototxt';
    solver_tran11_proto = 'wuhan/transfer_finetuning/11fe_solver_transfer_fixed.prototxt';
    weights_tran11_file = ['wuhan/transfer_finetuning/caffe_alexnet_fe_11train_iter_', num2str(iternum), '.caffemodel'];
end

gpu_id  = 0;
layer_rec = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7-ft', 'fc8-ft','fc9-ft'};
layer_rec_num = numel(layer_rec);

% experimetns;

%% Prepare data;
if exist('caffe-master/matlab/+caffe', 'dir')
  addpath('caffe-master/matlab');
else
  error('Please run this demo from deep_learning/LU');
end
addpath('feature_ext');

tic;

meandload = load(mean_file);
mean_data = meandload.mean_data;
mean_size = zeros(2,1);
[mean_size(1), mean_size(2), ~] = size(mean_data);

% DataInfo = ImageSet2Mat(lu_filepath, token,  crop_shape, argu_type, mean_data(:,:,[3,2,1]));
load([lu_filepath, '_', num2str(crop_shape(1)), 'crop_1times_mat_info.mat']);

feaSet = SearchFolder2Big( lu_filepath, token );
class_imgnum = feaSet.class_imgnum;
total_imgnum = sum(class_imgnum);
class_num = feaSet.class_num;
class_path  = feaSet.class_path;       % the path of each class
img_name    = feaSet.img_name ;        % contain the pathes for each image of each class
%% finetuning
disp('load images');
tic;
load( DataInfo.mat_path{1});
cur_imgnum = class_imgnum(1);
image_channels = size(CropArgData, 3);
imagedata = zeros(crop_shape(1), crop_shape(2), image_channels, total_imgnum, 'single');
imagedata(:,:,:,1:cur_imgnum) = CropArgData(:,:,:,:);
imid = cur_imgnum+1;
for is = 2 : class_num,
    load( DataInfo.mat_path{is});
    cur_imgnum = class_imgnum(is);
    imagedata(:,:,:,imid:imid+cur_imgnum-1) = CropArgData(:,:,:,:);
    imid = imid + cur_imgnum;
end
clear CropArgData;
toc;

tic;
caffe.set_mode_gpu();
caffe.set_device(gpu_id);
net = caffe.Net(model_alexnet_proto, weights_file, 'test'); % create net and load weights

cnn_feat = zeros(1, 1, 4096, total_imgnum, 'single');
disp('Extract cnn features');
for im = 1 : total_imgnum,
    data = imagedata(:,:,:,im);
    res = net.forward({data});
    cnn_feat(:,:,:,im) = res{1}; % net.blobs('fc6').get_data();
end
caffe.reset_all();
toc;

clear net res;
save([lu_filepath,'_', num2str(crop_shape(1)), 'crop_1times_cnnfeat.mat'], 'cnn_feat');

errormatrix = cell(rand_num,1);
oa_acc = zeros(rand_num,1);
record_num = floor(iternum/test_interval);
diff_wei_layers = zeros(record_num, layer_rec_num);

% sample_perm_all = zeros(rand_num, class_imgnum(1));
% for randi = 1 : rand_num,
%    sample_perm_all(randi,:) = randperm(class_imgnum(1));
% end
% save(rand_sample_file, 'sample_perm');
load(rand_sample_file);
for randi = 1 : rand_num,
    sample_perm = sample_perm_all(randi,:);
    
    disp('training mlp with cnn features...');
    tic;
    train_ids = zeros(class_imgnum(1), class_num);
    train_ids( sample_perm( 1 : floor( class_imgnum(1)* tr_samples_per ) ),  :) = 1;
    train_ids = train_ids(:);
    train_ids = train_ids';
    train_ids = logical(train_ids(:));
    num_train = sum(train_ids);
    num_train_pc = class_imgnum(1)* tr_samples_per;
    num_test = total_imgnum - num_train;
    num_test_pc = class_imgnum(1) - num_train_pc;
    
    sample_label = DataInfo.mat_label;
    order = 1:num_train;
    order = reshape(order, [num_train_pc, class_num]);
    order = permute(order, [2, 1]);
    order2 = 1:num_test;
    order2 = reshape(order2, [num_test_pc, class_num]);
    order2 = permute(order2, [2,1]);
    
    % training classifier
    tr_cnn_feat = cnn_feat(:,:,:,train_ids);
    tr_cnn_feat_label = sample_label(train_ids) - 1; % the start label is from 0;
    tr_cnn_feat = tr_cnn_feat(:,:,:,order(:));
    tr_cnn_feat_label = tr_cnn_feat_label(order(:));
    te_cnn_feat = cnn_feat(:,:,:,~train_ids);
    te_cnn_feat_label = sample_label(~train_ids) - 1;
    te_cnn_feat = te_cnn_feat(:,:,:,order2(:));
    te_cnn_feat_label = te_cnn_feat_label(order2(:));
    
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
    solver = caffe.Solver(solver_tran11_proto);
    net = solver.net;
    net.layers('data').set_memdata(tr_cnn_feat, tr_cnn_feat_label, num_train);
    test_net = solver.test_nets(1);
    test_net.layers('data').set_memdata(te_cnn_feat , te_cnn_feat_label, num_test);
    clear tr_cnn_feat te_cnn_feat;

    rec_loss1 = zeros(record_num,2); % training loss and test loss
    for ir = 1 : record_num,
        solver.step(test_interval);
        rec_loss1(ir, 1) = solver.net.blobs('accuracy').get_data();
        rec_loss1(ir, 2) = solver.test_nets(1).blobs('accuracy').get_data();
    end
    caffe.reset_all();
    toc;
    
    % finetuning
    disp('training mlp with cnn features...');
    tic;
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

    % full initialization finetuning
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
    solver = caffe.Solver(solver_proto);
    solver.net.copy_from(weights_file);
    solver.net.copy_from(weights_tran11_file);
    rec_loss2 = zeros(record_num,2); % training loss and test loss
    for ir = 1 : record_num,
        solver.step(test_interval);
        rec_loss2(ir, 1) = solver.net.blobs('accuracy').get_data();
        rec_loss2(ir, 2) = solver.test_nets(1).blobs('accuracy').get_data();
        if (rand_num == 1),
            for ilayer = 1: layer_rec_num,
                diff = solver.net.layers(layer_rec{ilayer}).params(1).get_diff();
                diff_wei_layers(ir, ilayer) = sum( abs(diff(:)) );
            end
        end      
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
            imageval = imresize(   imageval(:,:,[3,2,1]), [mean_size(1), mean_size(2)]);
            imageval = permute( single(imageval), [2,1,3])- mean_data;
            cropdata = imageval(CenW:CenW+crop_shape(1)-1, CenH:CenH+crop_shape(2)-1, :);
            net.forward({cropdata});
            test_pro(:,counttest) =  net.blobs('prob').get_data();
            te_label(counttest) = ic;
            counttest = counttest + 1;
        end
        count = count + class_imgnum(ic);
    end
    caffe.reset_all();

    [~, pred_labels] = max(test_pro);
    pred_labels = pred_labels(:);
    acc = assessment(int32(te_label(1:num_test)), int32(pred_labels),'class');
    fprintf('Round %d:  OA = %.2f%%\n', randi, acc.OA);
    errormatrix{randi} = acc;
    oa_acc(randi) = acc.OA;
    clear data net;
    toc;
end
total_time = toc;
fprintf('The total cost time (s) : %.1f\n', total_time);

save([lu_filepath, '_4tran_ft_full_' , num2str(crop_shape(1)), 'crop_acc.mat'],...
    'oa_acc', 'errormatrix', 'test_pro', 'rec_loss1', 'rec_loss2', 'diff_wei_layers', 'total_time');
fprintf('OA: mean = %.2f, std = %.2f (%%)\n', mean(oa_acc), std(oa_acc));

% load([lu_filepath, '_tran_finetuning_' , num2str(crop_shape(1)), 'crop_',  num2str(num_mul), 'times_acc.mat']);

rmpath('caffe-master/matlab');
rmpath('feature_ext');