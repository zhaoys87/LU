%% 
clc;
%% demo_cnn_social_data
% Demonstration for land-use classification
%% bei zhao, zhaoys@cuhk.edu.hk, Sep. 21st, 2016
%% input data
crop_shape = [227, 227];
test_interval = 100;
iternum = [10000, 5000];

type = 10; % 1 : image_alex_fe; 2: image_alex_ft; 3 : image small_nets; 4 fe + small_nets; 5 : social_cnn; 6 social_mlp;
%% 
          % 7: image_alex_ft + social_cnn; 8: small_nets + social_cnn;
          % 81-82: small_nets +social_cnn (feature level, 2,and decision level )
          % 9: (image_alex_fe + social_cnn) + small_nets; 10: image_alex_fe + (small_nets + social_cnn);
          % 12: (image_alex_fe + social_cnn) + (small_nets + social_cnn)

weights_alexnet_file = 'bvlc_alexnet/bvlc_alexnet.caffemodel';
fe_alexnet_proto = 'social-shenzhen/transfer_ft/fe_alexnet.prototxt';
fe_alexnet_classifier_test = 'social-shenzhen/transfer_ft/fe_alexnet_classifier_test.prototxt';
solver_fe_proto = 'social-shenzhen/transfer_ft/fe_solver_alexnet.prototxt';
weights_fe_file = ['social-shenzhen/transfer_ft/results/fe_alexnet_classifier','_iter_',num2str(iternum(1)) ,'.caffemodel'];

mean_file = 'social-shenzhen/sz-samples-mean.mat';
layer_rec = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7-ft', 'fc8-ft','fc9-ft'};
layer_rec_num = numel(layer_rec);
token = '*.jpg';

tr_sample_index = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/SZ-WorldView3/Social_Training_Samples';
val_sample_index = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/SZ-WorldView3/Social_Testing_Samples';
sample_image = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/SZ-WorldView3/Social_Sample_Images';

social_sample_image = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/SZ-WorldView3/Social_Data';
mean_social_file = 'social-shenzhen/sz-social-data-mean.mat';

if (type < 5),
    test_ft_proto = ['social-shenzhen/transfer_ft/ft', num2str(type), '_alex_test.prototxt'];
    solver_ft_proto = ['social-shenzhen/transfer_ft/ft', num2str(type), '_solver_alexnet.prototxt'];
    weights_ft_file = ['social-shenzhen/transfer_ft/results/ft', num2str(type), '_alexnet_iter_', ...
        num2str(iternum(1)), '.caffemodel'];
    weights_small_file = ['social-shenzhen/transfer_ft/results/ft3_alexnet_iter_', ...
            num2str(iternum(1)), '.caffemodel'];
elseif (type >= 5 && type < 7)
    test_ft_proto = ['social-shenzhen/social-only/', num2str(type), 'feat_socialdata.prototxt'];
    solver_ft_proto = ['social-shenzhen/social-only/', num2str(type), 'solver_socialdata.prototxt'];
    weights_ft_file = ['social-shenzhen/social-only/results/cnn_socialdata_', num2str(type), 'train_iter_', ...
        num2str(iternum(1)), '.caffemodel'];
    layer_social_rec = {'fc4-so', 'fc5-so'};
    layer_soc_rec_num = numel(layer_social_rec);
else
    test_ft_proto = ['social-shenzhen/union/ft', num2str(type), '_alex_test.prototxt'];
    solver_ft_proto = ['social-shenzhen/union/ft', num2str(type), '_solver_alexnet.prototxt'];
    weights_ft_file = ['social-shenzhen/union/results/ft', num2str(type), '_alexnet_iter_', ...
        num2str(iternum(1)), '.caffemodel'];
    weights_img_so_file = ['social-shenzhen/union/results/ft8_alexnet_iter_', ...
            num2str(iternum(1)), '.caffemodel'];
    weights_so_file = ['social-shenzhen/social-only/results/cnn_socialdata_5train_iter_', ...
            num2str(iternum(1)), '.caffemodel'];
    weights_alex_so_file = ['social-shenzhen/union/results/ft7_alexnet_iter_', ...
            num2str(iternum(1)), '.caffemodel'];
    weights_small_file = ['social-shenzhen/transfer_ft/results/ft3_alexnet_iter_', ...
            num2str(iternum(1)), '.caffemodel'];
    weights_alex_img_file = ['social-shenzhen/transfer_ft/results/ft4_alexnet_iter_', ...
            num2str(iternum(1)), '.caffemodel'];
end

rgb_cs = 3;
if(type == 1 || type == 2)
    model_channel = 3;
elseif(type == 5 || type == 6)
    model_channel = 1;
else
    model_channel = 4;
end
gpu_id  = 0;
% experimetns;
rand_num = 1;

%% Prepare data;
if exist('caffe-master/matlab/+caffe', 'dir')
  addpath('caffe-master/matlab');
else
  error('Please run this demo from deep_learning/LU');
end
addpath('feature_ext');

rec_time = zeros(3,1);

%% load mean data;
meandload = load(mean_file);
mean_data = meandload.mean_data;
mean_size = zeros(2,1);
[mean_size(1), mean_size(2), mean_dim] = size(mean_data);

meansocload = load(mean_social_file);
mean_social_data = meansocload.mean_data;
mean_social_size = size(mean_social_data);
%% extract the training image features;
disp('Extract features of the training images');
tic;

if(type == 1)
    DataInfo = HKWVImageSet2Mat(tr_sample_index, sample_image, token, mean_data,  crop_shape);
else
    load([tr_sample_index, '_mat_info.mat']);
end
class_imgnum1 = DataInfo.class_imgnum;
total_imgnum1 = sum(class_imgnum1);
class_num1 = DataInfo.class_num;
tr_cnn_feat_label = DataInfo.mat_label - 1;
if(type == 1)
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
    net = caffe.Net(fe_alexnet_proto, weights_alexnet_file, 'test'); % create net and load weights
    tr_cnn_feat = zeros(1, 1, 4096, total_imgnum1, 'single');
    imid = 1;
    for is = 1 : class_num1,
        load( DataInfo.mat_path{is});
        for ii = 1 : class_imgnum1(is),
           data = CropArgData(:,:,1:rgb_cs,ii);
           res = net.forward({data});
           tr_cnn_feat(:,:,:,imid) = res{1}; % net.blobs('fc6').get_data();    
           imid = imid + 1;
        end
    end
    clear CropArgData data res net;
    caffe.reset_all();
    save([tr_sample_index, '_cnnfeat.mat'], 'tr_cnn_feat');
else
    load([tr_sample_index, '_cnnfeat.mat']);
end

% extract the validation image features;
disp('Extract features of the training images');
if(type == 1)
    DataInfo = HKWVImageSet2Mat(val_sample_index, sample_image, token, mean_data,  crop_shape);
else
    load([val_sample_index, '_mat_info.mat']);
end
class_imgnum2 = DataInfo.class_imgnum;
total_imgnum2 = sum(class_imgnum2);
class_num2 = DataInfo.class_num;
image_name2 = DataInfo.img_name;
val_cnn_feat_label = DataInfo.mat_label - 1;
if(type == 1)
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
    net = caffe.Net(fe_alexnet_proto, weights_alexnet_file, 'test'); % create net and load weights
    val_cnn_feat = zeros(1, 1, 4096, total_imgnum2, 'single');
    imid = 1;
    for is = 1 : class_num2,
        load( DataInfo.mat_path{is});
        for ii = 1 : class_imgnum2(is),
           data = CropArgData(:,:,1:rgb_cs,ii);
           res = net.forward({data});
           val_cnn_feat(:,:,:,imid) = res{1}; % net.blobs('fc6').get_data();    
           imid = imid + 1;
        end
    end
    clear CropArgData data res net;
    caffe.reset_all();
    save([val_sample_index, '_cnnfeat.mat'], 'val_cnn_feat');
else
    load([val_sample_index, '_cnnfeat.mat']);
end

rec_time(1) = toc;

%% finetuning
errormatrix = cell(rand_num,1);
oa_acc = zeros(rand_num,1);
record_num = floor(iternum/test_interval);
rec_loss1 = zeros(record_num(1),2); % training loss and test loss
rec_loss2 = zeros(record_num(1),2); % training loss and test loss
if (rand_num == 1 && type == 1),
    diff_wei_layers = zeros(record_num(1), layer_rec_num);
elseif(rand_num == 5)
    diff_wei_layers = zeros(record_num(1), layer_soc_rec_num);
else
    diff_wei_layers = [];
end
order = 1 : total_imgnum1;
reorder = reshape(order, [class_imgnum1(1), class_num1]);
reorder = reorder';
reorder = reorder(:);
for randi = 1 : rand_num,
    disp('training mlp with cnn features...');
    tic;
    % training classifier
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
     if(type == 1)
        solver = caffe.Solver(solver_fe_proto);
        net = solver.net;
        net.layers('data').set_memdata(tr_cnn_feat(:,:,:,reorder), tr_cnn_feat_label(reorder), total_imgnum1);
        test_net = solver.test_nets(1);
        test_net.layers('data').set_memdata(val_cnn_feat , val_cnn_feat_label, total_imgnum2);
        % clear tr_cnn_feat te_cnn_feat;
        rec_loss1(:) = 0; % training loss and test loss
        for ir = 1 : record_num(1),
            solver.step(test_interval);
            rec_loss1(ir, 1) = solver.net.blobs('accuracy').get_data();
            rec_loss1(ir, 2) = solver.test_nets(1).blobs('accuracy').get_data();
        end
    else
        % finetuning
        disp('Fine tuning the cnn ...');
        solver = caffe.Solver(solver_ft_proto);
        if(type == 2 || type == 4 || type == 7 || type == 10 || type == 12)
            solver.net.copy_from(weights_alexnet_file);
            solver.net.copy_from(weights_fe_file);
        end
        if(type == 4 || type == 9)
            solver.net.copy_from(weights_small_file);
        end
        if(type == 9)
            solver.net.copy_from(weights_alex_so_file);
        end
        if(type == 10 || type == 12)
            solver.net.copy_from(weights_img_so_file);
        end
        if(type == 7 || type == 8)
            solver.net.copy_from(weights_so_file);
        end
        rec_loss2(:) = 0; % training loss and test loss
        for ir = 1 : record_num(1),
            solver.step(test_interval);
            rec_loss2(ir, 1) = solver.net.blobs('accuracy').get_data();
            rec_loss2(ir, 2) = solver.test_nets(1).blobs('accuracy').get_data();
            if (rand_num == 1 && type == 2 ),
                for ilayer = 1: layer_rec_num,
                    diff = solver.net.layers(layer_rec{ilayer}).params(1).get_diff();
                    diff_wei_layers(ir, ilayer) = sum( abs(diff(:)) );
                end
            elseif (rand_num == 1 && type == 5)
                for ilayer = 1: layer_soc_rec_num,
                    diff = solver.net.layers(layer_social_rec{ilayer}).params(1).get_diff();
                    diff_wei_layers(ir, ilayer) = sum( abs(diff(:)) );
                end
            end
        end
    end
    caffe.reset_all();
    rec_time(2) = toc;
    
    % get confusion matrix
    tic;
    caffe.set_device(gpu_id);
    if(type == 1)
        net = caffe.Net(fe_alexnet_classifier_test, weights_fe_file, 'test'); % create net and load weights
    else
        net = caffe.Net(test_ft_proto, weights_ft_file, 'test'); % create net and load weights
    end
    test_pro = zeros(class_num2, total_imgnum2, 'single');
    te_label = zeros(total_imgnum2, 1,'int32');
    count  = 1;
    if(type == 1)
        for ic = 1 : class_num2,
            curindices = count : count + class_imgnum2(ic) - 1;
            net.forward({val_cnn_feat(:,:,:,curindices)});
            test_pro(:,curindices) =  net.blobs('prob').get_data();
            te_label(curindices) = ic;
            count = count + class_imgnum2(ic);
        end
    elseif(type >= 5 && type < 7)
        for ic = 1 : class_num2,
            for iid = 1 : class_imgnum2(ic),
                [fpt,filename ] = fileparts(image_name2{ count});
                filename = fullfile(social_sample_image, filename);
                [socialval, curdim] = freadenvi(filename);
                socialval = single(reshape(socialval, curdim)) - mean_social_data;
                net.forward({socialval});
                test_pro(:,count) =  net.blobs('prob').get_data();
                te_label(count) = ic;
                count = count + 1;
            end
        end
    elseif(type >= 7)
        CenW = floor((mean_size(1) - crop_shape(1) + 1)/2) ;
        CenH = floor((mean_size(2) - crop_shape(2) + 1)/2) ;
        for ic = 1 : class_num2,
            for iid = 1 : class_imgnum2(ic),
                [fpt,filename ] = fileparts(image_name2{ count});
                filename = fullfile(sample_image, filename);
                [imageval, curdim] = freadenvi(filename);
                imageval = reshape(imageval, curdim);
                imageval =  single(imageval(:,:,1:mean_dim))- mean_data ;
                cropdata = imageval(CenW:CenW+crop_shape(1)-1, CenH:CenH+crop_shape(2)-1, 1:model_channel);
                inputdata = cell(2,1);
                inputdata{1} = cropdata;
                [fpt,filename ] = fileparts(image_name2{ count});
                filename = fullfile(social_sample_image, filename);
                [socialval, curdim] = freadenvi(filename);
                socialval = single(reshape(socialval, curdim)) - mean_social_data;
                inputdata{2} = socialval;
                net.forward(inputdata);
                test_pro(:,count) =  net.blobs('prob').get_data();
                te_label(count) = ic;
                count = count + 1;
            end
        end
    else
        CenW = floor((mean_size(1) - crop_shape(1) + 1)/2) ;
        CenH = floor((mean_size(2) - crop_shape(2) + 1)/2) ;
        for ic = 1 : class_num2,
            for iid = 1 : class_imgnum2(ic),
                [fpt,filename ] = fileparts(image_name2{ count});
                filename = fullfile(sample_image, filename);
                [imageval, curdim] = freadenvi(filename);
                imageval = reshape(imageval, curdim);
                imageval =  single(imageval(:,:,1:mean_dim))- mean_data ;
                cropdata = imageval(CenW:CenW+crop_shape(1)-1, CenH:CenH+crop_shape(2)-1, 1:model_channel);
                net.forward({cropdata});
                test_pro(:,count) =  net.blobs('prob').get_data();
                te_label(count) = ic;
                count = count + 1;
            end
        end
    end
    caffe.reset_all();
    [~, pred_labels] = max(test_pro);
    pred_labels = pred_labels(:);
    acc = assessment(int32(te_label), int32(pred_labels),'class');
    fprintf('Round %d:  OA = %.2f%%\n', randi, acc.OA);
    errormatrix{randi} = acc;
    oa_acc(randi) = acc.OA;
    clear data net;
    rec_time(3) = toc;
end
fprintf('The total cost time (s) : %.1f\n', sum(rec_time));
if( type == 1 ),
    save([val_sample_index, '_cnnfeat_acc_ver',num2str(type),'.mat'],  'rec_loss1', ...
        'diff_wei_layers', 'rec_time', 'errormatrix', 'oa_acc', 'te_label', 'pred_labels');
else
    save([val_sample_index, '_cnnfeat_acc_ver',num2str(type),'.mat'],  'rec_loss2', ...
        'rec_time', 'errormatrix', 'oa_acc', 'te_label', 'pred_labels' );
end

rmpath('caffe-master/matlab');
rmpath('feature_ext');
