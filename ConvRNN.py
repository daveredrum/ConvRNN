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
    def __init__(self, input_size, filter_size, filter_stride, 
        pool_size, pool_stride, name="ConvGRUCell"):
        self.input_size = input_size
        self.hidden_size = input_size[:-1] + [filter_size[-1]]
        self.filter_size = filter_size
        self.filter_stride = filter_stride
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.w_filter = self._init_filter(filter_size)
        self.u_filter = self._init_filter(filter_size[:2]+[filter_size[-1], filter_size[-1]])
        self.bias = self._init_bias(self.hidden_size)
            
    def __call__(self, inputs, states):
        h = states
        z = self._conv(inputs=inputs, 
            h=h, w=self.w_filter['z'], u=self.u_filter['z'], b=self.bias['z'])
        r = self._conv(inputs=inputs, 
            h=h, w=self.w_filter['r'], u=self.u_filter['r'], b=self.bias['r'])
        h_tilde = self._conv(inputs=inputs, 
            h=tf.multiply(r, h), w=self.w_filter['h'], u=self.u_filter['h'], b=self.bias['h'])
        h = tf.multiply(z, h) + tf.multiply(tf.ones(self.hidden_size) - z, h_tilde)
        return h, h
            
    def _init_filter(self, size):
        filter = {}
        filter['z'] = tf.Variable(tf.truncated_normal(shape=size, stddev=0.1))
        filter['r'] = tf.Variable(tf.truncated_normal(shape=size, stddev=0.1))
        filter['h'] = tf.Variable(tf.truncated_normal(shape=size, stddev=0.1))
        return filter

    def _init_bias(self, size):
        bias = {}
        bias['z'] = tf.Variable(tf.constant(0.1, shape=size))
        bias['r'] = tf.Variable(tf.constant(0.1, shape=size))
        bias['h'] = tf.Variable(tf.constant(0.1, shape=size))
        return bias

    def _conv(self, inputs, h, w, u, b):
        w_conv = tf.nn.conv2d(inputs, w, self.filter_stride, padding="SAME")
        u_conv = tf.nn.conv2d(h, u, self.filter_stride, padding="SAME")
        out = w_conv + u_conv + b
        return tf.sigmoid(out)

    def initState(self):
        return tf.zeros(self.hidden_size)
