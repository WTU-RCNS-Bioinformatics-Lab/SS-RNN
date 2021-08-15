import torch
import torch.nn.functional as F
import math
from torch.nn import Parameter
import pdb
import numpy as np
import torch.nn as nn
from queue import Queue


class LSTM(nn.Module):

    def __init__(self, cell, input_size: int, hidden_size: int, bias: bool = True, batch_first=False):

        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first

        self.cell = cell


        gate_size = 4 * hidden_size

        w_ih = Parameter(torch.empty(gate_size, input_size))
        w_hh = Parameter(torch.empty(gate_size, hidden_size))
        b_ih = Parameter(torch.empty(gate_size))
        b_hh = Parameter(torch.empty(gate_size))
        params = (w_ih, w_hh, b_ih, b_hh)
        param_names = ['weight_ih', 'weight_hh']
        self._params = params

        if bias is True:
            param_names += ['bias_ih', 'bias_hh']

        for name, param in zip(param_names, params):
            setattr(self, name, param)

        self._flat_weights_names = param_names
        self._all_weights = param_names

        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in
                              self._flat_weights_names]
        self.flatten_parameters()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        # Short-circuits if _flat_weights is only partially instantiated
        if len(self._flat_weights) != len(self._flat_weights_names):
            return

        for w in self._flat_weights:
            if not torch.is_tensor(w):
                return
        # Short-circuits if any tensor in self._flat_weights is not acceptable to cuDNN
        # or the tensors in _flat_weights are of different dtypes

        first_fw = self._flat_weights[0]
        dtype = first_fw.dtype
        for fw in self._flat_weights:
            if (not torch.is_tensor(fw.data) or not (fw.data.dtype == dtype) or
                    not fw.data.is_cuda or
                    not torch.backends.cudnn.is_acceptable(fw.data)):
                return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        unique_data_ptrs = set(p.data_ptr() for p in self._flat_weights)
        if len(unique_data_ptrs) != len(self._flat_weights):
            return

        with torch.cuda.device_of(first_fw):
            import torch.backends.cudnn.rnn as rnn

            # Note: no_grad() is necessary since _cudnn_rnn_flatten_weight is
            # an inplace operation on self._flat_weights
            with torch.no_grad():
                if torch._use_cudnn_rnn_flatten_weight():
                    torch._cudnn_rnn_flatten_weight(
                        self._flat_weights, (4 if self.bias else 2),
                        self.input_size, rnn.get_cudnn_mode(self.mode), self.hidden_size, 1,
                        False, False)

    def get_expected_hidden_size(self, input, batch_sizes):
        # type: (Tensor, Optional[Tensor]) -> Tuple[int, int, int]

        mini_batch = input.size(1)

        expected_hidden_size = (mini_batch, self.hidden_size)
        return expected_hidden_size

    def check_input(self, input, batch_sizes):
        # type: (Tensor, Optional[Tensor]) -> None
        expected_input_dim = 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        if self.input_size != input.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(
                    self.input_size, input.size(-1)))

    def check_forward_args(self, input, hidden, batch_sizes):
        # type: (Tensor, Tensor, Optional[Tensor]) -> None
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden[0], expected_hidden_size)
        self.check_hidden_size(hidden[1], expected_hidden_size)

    def check_hidden_size(self, hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
        # type: (Tensor, Tuple[int, int, int], str) -> None
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

    def forward(self, input, hx=None):
        input_size = input.size()
        if self.batch_first is True:
            batch_size, time, input_dim = input_size
            input = input.transpose(1, 0)
        else:
            time, batch_size, input_dim = input_size

        if hx is None:
            zeros = torch.zeros(batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        self.check_forward_args(input, hx, batch_size)
        result = self.cell(input, hx, self._flat_weights, self.bias)

        output = result[0]
        hidden = result[1:]
        if self.batch_first is True:
            output = output.transpose(1, 0)

        return output, hidden


class LSTMCell(nn.Module):
    def __init__(self, **kwargs):
        super(LSTMCell, self).__init__()

    def forward(self, input, hx, weights, bias):
        hidden, candidate = hx
        integrated_weight = torch.cat((weights[0], weights[1]), 1)
        if bias:
            integrated_bias = weights[2] + weights[3]
        else:
            integrated_bias = None
        output = []
        for x in input:
            integrated_x = torch.cat((x, hidden), 1)
            gates = F.linear(integrated_x, integrated_weight, integrated_bias)

            input_gate, forget_gate, cell_state, output_gate = torch.chunk(gates, 4, 1)

            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            cell_state = torch.tanh(cell_state)
            output_gate = torch.sigmoid(output_gate)
            candidate = torch.add(forget_gate * candidate, input_gate * cell_state)
            hidden = output_gate * torch.tanh(candidate)
            output.append(hidden)

        output = torch.stack(output)

        return (output, hidden, candidate)


class SkipACell(torch.nn.Module):
    def __init__(self, **kwargs):
        super(SkipACell, self).__init__()
        self.skip = kwargs['skip']

    def forward(self, input, hx, weights, bias):

        hidden, candidate = hx
        # hidden_size = hidden.size(1)
        integrated_weight = torch.cat((weights[0], weights[1]), 1)
        if bias:
            integrated_bias = weights[2] + weights[3]
        else:
            integrated_bias = None
        output = []
        skip_connection = hidden
        skip_step = self.skip
        for x in input:
            integrated_x = torch.cat((x, hidden), 1)
            gates = F.linear(integrated_x, integrated_weight, integrated_bias)
            input_gate, forget_gate, cell_state, output_gate = torch.chunk(gates, 4, 1)
            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            cell_state = torch.tanh(cell_state)
            output_gate = torch.sigmoid(output_gate)
            candidate = torch.add(forget_gate * candidate, input_gate * cell_state)
            hidden = output_gate * torch.tanh(candidate)
            if skip_step == 0:
                hidden = hidden + skip_connection
                skip_connection = hidden
                skip_step = self.skip
            else:
                skip_step -= 1

            output.append(hidden)

        output = torch.stack(output)

        return (output, hidden, candidate)


class SkipBCell(nn.Module):
    def __init__(self, **kwargs):
        super(SkipBCell, self).__init__()
        self.skip = kwargs['skip']


    def forward(self, input, hx, weights, bias):
        weight_h_skip = None
        try:
            weight_h_skip = getattr(self, 'weight_h_skip')
        except:
            weight_hh_shape = weights[1].shape
            weight_h_skip = Parameter(torch.empty(weight_hh_shape).type_as(input))
            hidden_size = weights[1].shape[1]
            stdv = 1.0 / math.sqrt(hidden_size)
            weight_h_skip.data.uniform_(-stdv, stdv)
            setattr(self, 'weight_h_skip', weight_h_skip)

        hidden, candidate = hx
        integrated_weight = torch.cat((weights[0], weights[1]), 1)
        if bias:
            integrated_bias = weights[2] + weights[3]
        else:
            integrated_bias = None
        output = []
        skip_hidden = hidden
        skip = 0
        for x in input:
            integrated_x = torch.cat((x, hidden), 1)

            gates = F.linear(integrated_x, integrated_weight, integrated_bias)

            if skip == self.skip:
                gates = torch.add(gates, skip_hidden.matmul(weight_h_skip.t()))


            input_gate, forget_gate, cell_state, output_gate = torch.chunk(gates, 4, 1)

            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            cell_state = torch.tanh(cell_state)
            output_gate = torch.sigmoid(output_gate)
            candidate = torch.add(forget_gate * candidate, input_gate * cell_state)
            hidden = output_gate * torch.tanh(candidate)

            if skip == self.skip:
                skip_hidden = hidden
                skip = 0
            else:
                skip = skip + 1

            output.append(hidden)


        output = torch.stack(output)

        return (output, hidden, candidate)


class SkipCCell(nn.Module):
    def __init__(self, **kwargs):
        super(SkipCCell, self).__init__()
        self.skip = kwargs['skip']


    def forward(self, input, hx, weights, bias):
        hidden, candidate = hx
        integrated_weight = torch.cat((weights[0], weights[1]), 1)
        if bias:
            integrated_bias = weights[2] + weights[3]
        else:
            integrated_bias = None
        output = []
        skip_hidden_queue  = Queue(maxsize=self.skip)

        for x in input:
            integrated_x = torch.cat((x, hidden), 1)
            gates = F.linear(integrated_x, integrated_weight, integrated_bias)

            input_gate, forget_gate, cell_state, output_gate = torch.chunk(gates, 4, 1)

            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            cell_state = torch.tanh(cell_state)
            output_gate = torch.sigmoid(output_gate)
            candidate = torch.add(forget_gate * candidate, input_gate * cell_state)
            hidden = output_gate * torch.tanh(candidate)

            if skip_hidden_queue.full():
                hidden = hidden + skip_hidden_queue.get()

            skip_hidden_queue.put(hidden)
            output.append(hidden)

        output = torch.stack(output)

        return (output, hidden, candidate)


class SkipDCell(nn.Module):
    def __init__(self, **kwargs):
        super(SkipDCell, self).__init__()
        self.skip = kwargs['skip']

    def forward(self, input, hx, weights, bias):

        weight_h_skip = None
        try:
            weight_h_skip = getattr(self, 'weight_h_skip')
        except:
            weight_hh_shape = weights[1].shape
            weight_h_skip = Parameter(torch.empty(weight_hh_shape).type_as(input))
            hidden_size = weights[1].shape[1]
            stdv = 1.0 / math.sqrt(hidden_size)
            weight_h_skip.data.uniform_(-stdv, stdv)
            setattr(self, 'weight_h_skip', weight_h_skip)

        hidden, candidate = hx
        integrated_weight = torch.cat((weights[0], weights[1]), 1)
        if bias:
            integrated_bias = weights[2] + weights[3]
        else:
            integrated_bias = None
        output = []
        skip_hidden_queue = Queue(maxsize=self.skip)
        for x in input:
            integrated_x = torch.cat((x, hidden), 1)
            gates = F.linear(integrated_x, integrated_weight, integrated_bias)
            if skip_hidden_queue.full():
                skip_hidden = skip_hidden_queue.get()
                gates = torch.add(gates, skip_hidden.matmul(weight_h_skip.t()))
            input_gate, forget_gate, cell_state, output_gate = torch.chunk(gates, 4, 1)

            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            cell_state = torch.tanh(cell_state)
            output_gate = torch.sigmoid(output_gate)
            candidate = torch.add(forget_gate * candidate, input_gate * cell_state)
            hidden = output_gate * torch.tanh(candidate)
            skip_hidden_queue.put(hidden)
            output.append(hidden)

        output = torch.stack(output)

        return (output, hidden, candidate)



class SkipECell(nn.Module):
    def __init__(self, **kwargs):
        super(SkipECell, self).__init__()
        self.skip = kwargs['skip']

    def forward(self, input, hx, weights, bias):
        weight_h_skip_name_list = ['weight_h_skip_{}'.format(i+1) for i in range(self.skip - 1)]
        weight_h_skip_list = []
        try:
            weight_h_skip_list = [getattr(self, weight_name) for weight_name in weight_h_skip_name_list]
        except:
            weight_hh_shape = weights[1].shape
            hidden_size = weights[1].shape[1]
            stdv = 1.0 / math.sqrt(hidden_size)
            for i in range(self.skip - 1):
                weight_h_skip = Parameter(torch.empty(weight_hh_shape).type_as(input))
                weight_h_skip.data.uniform_(-stdv, stdv)
                weight_h_skip_list.append(weight_h_skip)

            for name,weight in zip(weight_h_skip_name_list, weight_h_skip_list):
                setattr(self, name, weight)

        hidden, candidate = hx
        integrated_weight = torch.cat((weights[0], weights[1]), 1)
        if bias:
            integrated_bias = weights[2] + weights[3]
        else:
            integrated_bias = None
        output = []
        skip_hidden_queue = Queue(maxsize=self.skip)
        for x in input:
            integrated_x = torch.cat((x, hidden), 1)
            gates = F.linear(integrated_x, integrated_weight, integrated_bias)

            if skip_hidden_queue.full():
                queue_size = skip_hidden_queue.qsize()
                for i in range(queue_size):
                    if i == queue_size-1:
                        continue
                    skip_hidden = skip_hidden_queue.get()
                    weight_h_skip = weight_h_skip_list[i]
                    gates = torch.add(gates, skip_hidden.matmul(weight_h_skip.t()))


            input_gate, forget_gate, cell_state, output_gate = torch.chunk(gates, 4, 1)

            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            cell_state = torch.tanh(cell_state)
            output_gate = torch.sigmoid(output_gate)
            candidate = torch.add(forget_gate * candidate, input_gate * cell_state)
            hidden = output_gate * torch.tanh(candidate)
            skip_hidden_queue.put(hidden)
            output.append(hidden)

        output = torch.stack(output)

        return (output, hidden, candidate)


class SkipFCell(nn.Module):
    def __init__(self, **kwargs):
        super(SkipFCell, self).__init__()
        self.skip = kwargs['skip']

    def forward(self, input, hx, weights, bias):
        hidden, candidate = hx
        integrated_weight = torch.cat((weights[0], weights[1]), 1)
        if bias:
            integrated_bias = weights[2] + weights[3]
        else:
            integrated_bias = None
        output = []
        skip_hidden_queue = Queue(maxsize=self.skip)
        for x in input:
            integrated_x = torch.cat((x, hidden), 1)
            gates = F.linear(integrated_x, integrated_weight, integrated_bias)

            input_gate, forget_gate, cell_state, output_gate = torch.chunk(gates, 4, 1)

            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            cell_state = torch.tanh(cell_state)
            output_gate = torch.sigmoid(output_gate)
            candidate = torch.add(forget_gate * candidate, input_gate * cell_state)
            hidden = output_gate * torch.tanh(candidate)

            if skip_hidden_queue.full():
                queue_size = skip_hidden_queue.qsize()
                for i in range(queue_size):
                    skip_hidden = skip_hidden_queue.get()
                    if i == queue_size - 1:
                        continue
                    else:
                        hidden = hidden + skip_hidden
            else:
                skip_hidden_queue.put(hidden)
            output.append(hidden)

        output = torch.stack(output)

        return (output, hidden, candidate)


if __name__ == '__main__':
    batch_size = 34
    seq_length = 180
    dim = 5
    device = torch.device('cuda')
    net = LSTM(cell=SkipECell(skip=5), input_size=dim, hidden_size=6, batch_first=False)
    net.to(device)
    inputs = torch.randn((seq_length, batch_size, dim))
    inputs = inputs.to(device)
    for i in range(1):
        outputs, (h_n, c_n) = net(inputs)

    # print(inputs.shape, outputs.shape)

    for param in net.parameters():
        print(param.shape)
