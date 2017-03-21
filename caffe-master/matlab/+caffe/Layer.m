classdef Layer < handle
  % Wrapper class of caffe::Layer in matlab
  
  properties (Access = private)
    hLayer_self
    attributes
    % attributes fields:
    %     hBlob_blobs
  end
  properties (SetAccess = private)
    params
  end
  
  methods
    function self = Layer(hLayer_layer)
      CHECK(is_valid_handle(hLayer_layer), 'invalid Layer handle');
      
      % setup self handle and attributes
      self.hLayer_self = hLayer_layer;
      self.attributes = caffe_('layer_get_attr', self.hLayer_self);
      
      % setup weights
      self.params = caffe.Blob.empty();
      for n = 1:length(self.attributes.hBlob_blobs)
        self.params(n) = caffe.Blob(self.attributes.hBlob_blobs(n));
      end
    end
    function layer_type = type(self)
      layer_type = caffe_('layer_get_type', self.hLayer_self);
    end
    %% zhaoys, Mar 28, 2016;
    function set_memdata(self, data, label, n)
      layer_type = caffe_('layer_get_type', self.hLayer_self);
      CHECK(strcmp(layer_type, 'MemoryData'), 'layer must be a MemoryData layer');
      CHECK(isnumeric(data), 'data must be numeric types');
      CHECK(isnumeric(label), 'label must be numeric types');
      CHECK(isnumeric(n), 'n must be numeric types');
      if ~isa(data, 'single')
        data = single(data);
      end
      if ~isa(label, 'single')
        label = single(label);
      end
      if ~isa(n, 'double')
        n = double(n);
      end
      caffe_('layer_set_memdata', self.hLayer_self, data, label, n);
    end
  end
end
