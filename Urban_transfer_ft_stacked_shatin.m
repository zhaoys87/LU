clc; 
%% demo_transfer_ft_stacked_shatin
% Demonstration for land-use classification
%% bei zhao, zhaoys@cuhk.edu.hk, March 21, 2016
%% input data

crop_shape = [227, 227];
test_interval = 100;
iternum = [10000, 5000];

type = 4; % 1 : fine tuning RGB; 2: training alexnet; 3 : small nets full channels; 4 coupled dcnn; 

weights_alexnet_file = 'bvlc_alexnet/bvlc_alexnet.caffemodel';
fe_alexnet_proto = 'shatin-wv3/transfer_ft/fe_alexnet.prototxt';
solver_fe_proto = 'shatin-wv3/transfer_ft/fe_solver_alexnet.prototxt';
weights_fe_file = ['shatin-wv3/transfer_ft/results/fe_alexnet_classifier','_iter_',num2str(iternum(1)) ,'.caffemodel'];

test_ft_proto = ['shatin-wv3/transfer_ft/ft', num2str(type), '_alex_test.prototxt'];
solver_ft_proto = ['shatin-wv3/transfer_ft/ft', num2str(type), '_solver_alexnet.prototxt'];
weights_ft_file = ['shatin-wv3/transfer_ft/results/ft', num2str(type), '_alexnet_iter_', ...
    num2str(iternum(1)), '.caffemodel'];

if(type == 4)
weights_small_file = ['shatin-wv3/transfer_ft/results/ft', num2str(3), '_alexnet_iter_', ...
    num2str(iternum(1)), '.caffemodel'];
weights_ft1_file = ['shatin-wv3/transfer_ft/results/ft', num2str(1), '_alexnet_iter_', ...
    num2str(iternum(1)), '.caffemodel'];
end

tr_sample_index = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/HK-WV3-Shatin/Training_Samples';
val_sample_index = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/HK-WV3-Shatin/Testing_Samples';
sample_image = '/home/zhaoys/Documents/deep_learning/Scene_HSRI/HK-WV3-Shatin/Sample_Images';

mean_file = 'shatin-wv3/shatin-samples-mean.mat';
layer_rec = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7-ft', 'fc8-ft','fc9-ft'};
layer_rec_num = numel(layer_rec);
token = '*.jpg';

rgb_cs = 3;
if(type == 1 || type == 2)
    model_channel = 3;
else
    model_channel = 8;
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

rec_time = zeros(4,1);

%% load mean data;
meandload = load(mean_file);
mean_data = meandload.mean_data;
mean_size = zeros(2,1);
[mean_size(1), mean_size(2), mean_dim] = size(mean_data);

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
diff_wei_layers = zeros(record_num(1), layer_rec_num);
order = 1 : total_imgnum1;
reorder = reshape(order, [class_imgnum1(1), class_num1]);
reorder = reorder';
reorder = reorder(:);
for randi = 1 : rand_num,
    disp('training mlp with cnn features...');
    tic;
    
    % training classifier
    if(type == 1)
        caffe.set_mode_gpu();
        caffe.set_device(gpu_id);
        solver = caffe.Solver(solver_fe_proto);
        net = solver.net;
        net.layers('data').set_memdata(tr_cnn_feat(:,:,:,reorder), tr_cnn_feat_label(reorder), total_imgnum1);
        test_net = solver.test_nets(1);
        test_net.layers('data').set_memdata(val_cnn_feat , val_cnn_feat_label, total_imgnum2);
        % clear tr_cnn_feat te_cnn_feat;
        rec_loss1 = zeros(record_num(1),2); % training loss and test loss
        for ir = 1 : record_num(1),
            solver.step(test_interval);
            rec_loss1(ir, 1) = solver.net.blobs('accuracy').get_data();
            rec_loss1(ir, 2) = solver.test_nets(1).blobs('accuracy').get_data();
        end
        caffe.reset_all();
    end
    rec_time(2) = toc;
    
    % finetuning
    disp('Fine tuning the cnn ...');
    tic;
    
    caffe.set_mode_gpu();
    caffe.set_device(gpu_id);
    solver = caffe.Solver(solver_ft_proto);
    if(type == 1)
        solver.net.copy_from(weights_alexnet_file);
        solver.net.copy_from(weights_fe_file);
    else if(type == 4)
            solver.net.copy_from(weights_alexnet_file);
            solver.net.copy_from(weights_fe_file);            
            solver.net.copy_from(weights_small_file);
        end
    end
    rec_loss2 = zeros(record_num(1),2); % training loss and test loss
    for ir = 1 : record_num(1),
        solver.step(test_interval);
        rec_loss2(ir, 1) = solver.net.blobs('accuracy').get_data();
        rec_loss2(ir, 2) = solver.test_nets(1).blobs('accuracy').get_data();
        if (rand_num == 1 && type == 1 ),
            for ilayer = 1: layer_rec_num,
                diff = solver.net.layers(layer_rec{ilayer}).params(1).get_diff();
                diff_wei_layers(ir, ilayer) = sum( abs(diff(:)) );
            end
        end      
    end
    caffe.reset_all();
    rec_time(3) = toc;
    
    % get confusion matrix
    tic;
    
    caffe.set_device(gpu_id);
    net = caffe.Net(test_ft_proto, weights_ft_file, 'test'); % create net and load weights
    test_pro = zeros(class_num2, total_imgnum2, 'single');
    te_label = zeros(total_imgnum2, 1,'int32');
    count  = 1;
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
    caffe.reset_all();
    [~, pred_labels] = max(test_pro);
    pred_labels = pred_labels(:);
    acc = assessment(int32(te_label), int32(pred_labels),'class');
    fprintf('Round %d:  OA = %.2f%%\n', randi, acc.OA);
    errormatrix{randi} = acc;
    oa_acc(randi) = acc.OA;
    clear data net;
    
    rec_time(4) = toc;
end
fprintf('The total cost time (s) : %.1f\n', sum(rec_time));
if( type == 1 ),
    save([val_sample_index, '_cnnfeat_acc_ver',num2str(type),'.mat'],  'rec_loss1', 'rec_loss2', ...
        'diff_wei_layers', 'rec_time', 'errormatrix', 'oa_acc', 'te_label', 'pred_labels');
else
    save([val_sample_index, '_cnnfeat_acc_ver',num2str(type),'.mat'],  'rec_loss2', ...
        'rec_time', 'errormatrix', 'oa_acc', 'te_label', 'pred_labels' );
end

rmpath('caffe-master/matlab');
rmpath('feature_ext');