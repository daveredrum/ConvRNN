import tensorflow as tf

class ConvLSTM:
    def __init__(self, input_size, filter_size, filter_stride, 
        pool_size, pool_stride, name="ConvLSTMCell"):
        self.input_size = input_size
        self.hidden_size = input_size[:-1] + [filter_size[-1]]
        self.filter_size = filter_size
        self.filter_stride = filter_stride
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.w_filter = self._init_filter(filter_size)
        self.u_filter = self._init_filter(filter_size[:2]+[filter_size[-1], filter_size[-1]])
        self.v_filter = self._init_filter(self.hidden_size)
        self.bias = self._init_bias(self.hidden_size)

    def __call__(self, inputs, states):
        c, h = tf.split(states, 2, 3)
        f = self._conv(inputs=inputs, 
            c=c, h=h, w=self.w_filter['f'], u=self.u_filter['f'], v=self.v_filter['f'], b=self.bias['f'])
        i = self._conv(inputs=inputs, 
            c=c, h=h, w=self.w_filter['i'], u=self.u_filter['i'], v=self.v_filter['i'], b=self.bias['i'])
        o = self._conv(inputs=inputs, 
            c=c, h=h, w=self.w_filter['o'], u=self.u_filter['o'], v=self.v_filter['o'], b=self.bias['o'])
        c_tilde = tf.sigmoid(self._conv(inputs=inputs,
            c=c, h=h, w=self.w_filter['c'], u=self.u_filter['c'], v=None, b=self.bias['c']))
        c = tf.multiply(c, f) + tf.multiply(i, c_tilde)
        h = tf.multiply(o, tf.tanh(c))
        return h, tf.concat([c, h], 3)

    def _init_filter(self, size):
        filter = {}
        filter['f'] = tf.Variable(tf.truncated_normal(shape=size, stddev=0.1))
        filter['i'] = tf.Variable(tf.truncated_normal(shape=size, stddev=0.1))
        filter['c'] = tf.Variable(tf.truncated_normal(shape=size, stddev=0.1))
        filter['o'] = tf.Variable(tf.truncated_normal(shape=size, stddev=0.1))
        return filter

    def _init_bias(self, size):
        bias = {}
        bias['f'] = tf.Variable(tf.constant(0.1, shape=size))
        bias['i'] = tf.Variable(tf.constant(0.1, shape=size))
        bias['c'] = tf.Variable(tf.constant(0.1, shape=size))
        bias['o'] = tf.Variable(tf.constant(0.1, shape=size))
        return bias

    def _conv(self, inputs, c, h, w, u, v, b):
        w_conv = tf.nn.conv2d(inputs, w, self.filter_stride, padding="SAME")
        u_conv = tf.nn.conv2d(h, u, self.filter_stride, padding="SAME")
        if v:
            v_ew = tf.multiply(v, c)
            out = w_conv + u_conv + v_ew + b
        else:
            out = w_conv + u_conv + b
        return tf.sigmoid(out)

    def initState(self):
        return tf.concat([tf.zeros(self.hidden_size), tf.zeros(self.hidden_size)], 3)



class ConvGRU:
    def __init__(self, input_size, filter_size, filter_stride, pool_size, pool_stride, name="ConvGRUCell"):
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
        self.gru_variable = self._initialize_lstm()
            
    def __call__(self, inputs, state):
        with tf.variable_scope(self.name):
            conv_output = self._conv(inputs)
            conv_shape = conv_output.get_shape().as_list()
            flattened = tf.reshape(conv_output, [conv_shape[0], -1])
            outputs, state = self._gru(flattened, state)
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
    
    def _gru(self, inputs, state):
        h = state
        z = tf.sigmoid(tf.matmul((h + inputs), self.gru_variable['wz']))
        r = tf.sigmoid(tf.matmul((h + inputs), self.gru_variable['wr']))
        h_tilde = tf.tanh(tf.multiply(self.gru_variable['w'], (r * h) + inputs))
        h = (1 - z) * h + z * h_tilde
        return h, h
    
    def initState(self):
        return tf.zeros([self.input_size[0], self.conv_output])
    
    def _initialize_lstm(self):
        variables = {}
        variables['wz'] = tf.Variable(tf.truncated_normal([self.conv_output, self.input_size[0]], stddev=0.1))
        variables['wr'] = tf.Variable(tf.truncated_normal([self.conv_output, self.input_size[0]], stddev=0.1))
        variables['w'] = tf.Variable(tf.truncated_normal([self.input_size[0], self.conv_output], stddev=0.1))
        return variables