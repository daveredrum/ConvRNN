ass ConvLSTM:
    def __init__(self, input_size, filter_size, filter_stride, pool_size, pool_stride, name="ConvLSTMCell"):
        # filter_size is a dict of which the keys are layer numbers and 
        # the values are 4-D lists
        self.input_size = input_size
        self.filter_size = filter_size
        self.filter_stride = filter_stride
        self.conv_layers = len(filter_size.keys())
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.name = name
        self.weight = {}
        self.bias = {}
        _width = self.input_size[1]
        _height = self.input_size[2]
        for key in filter_size.keys():
            self.weight[key] = self._weight_variable(filter_size[key])
            self.bias[key] = self._bias_variable([filter_size[key][-1]])
            _width =  _width // (self.filter_stride[1] * self.pool_stride[1])
            _height = _height // (self.filter_stride[2] * self.pool_stride[2])
        self.conv_output = _width * _height * self.filter_size[self.conv_layers][-1]
        self.lstm = tf.contrib.rnn.BasicLSTMCell(self.conv_output)
            
    def __call__(self, inputs, state):
        with tf.variable_scope(self.name):
            conv_output = self._conv(inputs)
            conv_shape = conv_output.get_shape().as_list()
            flattened = tf.reshape(conv_output, [conv_shape[0], -1])
            outputs, state = self.lstm(flattened, state)
            return outputs, state
            
    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def _conv(self, inputs):
        for layer in self.filter_size.keys():
            inputs = tf.nn.conv2d(
                inputs, self.weight[layer], self.filter_stride, padding="SAME") + self.bias[layer]
            inputs = tf.nn.relu(inputs)
            inputs = tf.nn.max_pool(inputs, self.pool_size, self.pool_stride, padding="SAME")
        return inputs
