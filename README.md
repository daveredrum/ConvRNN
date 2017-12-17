# ConvRNN
Implementations of convolutional recurrent neural cells

## Usage
```python
# define the filters as dict
filter_size = {1: [7, 7, 1, 6], 2: [7, 7, 6, 12]}
convlstm = ConvLSTM(filter_size=filter_size, 
                    filter_stride=[1, 1, 1, 1], 
                    pool_size=[1, 2, 2, 1], 
                    pool_stride=[1, 2, 2, 1])
# initial call
state = convlstm.initState()
outputs, state = convlstm(inputs, state)
```
