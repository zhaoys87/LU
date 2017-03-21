1. add interface of MemoryDataLayer::Setdata(Dtype* pdata, Dtype* plabel, int n);
(1) matlab/+caffe/private/caffe_.cpp
	add interface : caffe_('layer_set_memdata', hlayer, data, label, n)
	compile matcaffe
(2) matlab/+caffe/Layer.m
	add member function : set_memdata

 memorydatalayer segmentation violation,
	math_functions.cu:121] Check failed: status == CUBLAS_STATUS_SUCCESS (14 vs. 0) CUBLAS_STATUS_INTERNAL_ERROR
https://github.com/TJKlein/caffe/commit/5f1bb97a587043dbe0892466b866abfe4c76804c
(1) src/caffe/solvers/sgd_solver.cpp

2. error and weight trace
https://github.com/BVLC/caffe/pull/2327/files
(1) src/caffe/proto/caffe.proto
(2) include/caffe/solver.hpp;
(3) src/caffe/solver.cpp
(4) tools/caffe.cpp


3. add min pooling operator
(1) src/caffe/proto/caffe.proto
(2) include/caffe/layers/pooling_layer.hpp;
(3) src/caffe/solvers/pooling_layer.cu;
(4) src/caffe/solvers/pooling_layer.cpp;

4. add my image read layers
(1) src/caffe/proto/caffe.proto	;	
(2) add include/caffe/layers/rsimage_data_layer.hpp;
(3) add src/caffe/layers/rsimage_data_layer.cpp;
(4) include/caffe/util/io.hpp;
(5) src/caffe/util/io.cpp;
(6) add include/caffe/util/XImage.hpp;
(7) add src/caffe/util/XImage.cpp;

