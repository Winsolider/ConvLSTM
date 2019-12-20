import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    # input: (batch,input_channels,height,width)
    # Parameters:
    #             input_channels - the number of expected feature in the input
    #             hidden_channels - the number of output channels
    #             kernel_size - the kernel size used in convolution
    #             nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'

    def __init__(self, input_channels, hidden_channels, kernel_size, nonlinearity):
        super(ConvLSTMCell, self).__init__()

        #assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.nonlinearity = nonlinearity        
        self.relu = nn.ReLU(inplace=True)

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        #self.Wxf.bias = Variable(torch.ones(self.input_channels, self.hidden_channels, self.kernel_size, self.kernel_size)).cuda()

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        if self.nonlinearity=='relu':
            cc = cf * c + ci * self.relu(self.Wxc(x) + self.Whc(h))
        else:
            cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        if self.nonlinearity=='relu':
            ch = co * self.relu(cc)
        else:
            ch = co * torch.tanh(cc)

        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):
    # input: (batch,input_channels,height,width)
    # Parameters:
    #             input_channels - the number of expected feature in the input
    #             hidden_channels - a list, include all succeeding layers hidden channels
    #             kernel_size - the kernel size used in convolution
    #             nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'

    def __init__(self, input_channels, hidden_channels, kernel_size, nonlinearity='tanh'):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.nonlinearity=nonlinearity
        self._all_layers = []
        self.internal_state_uninited = True
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.nonlinearity)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, x, state):
        if state is None:
            internal_state = []
        else:
            internal_state = state
        for i in range(self.num_layers):
            # all cells are initialized in the first step
            name = 'cell{}'.format(i)
            if state is None:
                bsize, _, height, width = x.size()
                (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                internal_state.append((h, c))

            # do forward
            (h, c) = internal_state[i]
            x, new_c = getattr(self, name)(x, h, c)
            internal_state[i] = (x, new_c)

        return x, internal_state


if __name__ == '__main__':
    # gradient check
    convlstm = ConvLSTM(input_channels=20, hidden_channels=[64, 32, 20], kernel_size=3, nonlinearity='relu').cuda()
    loss_fn = torch.nn.MSELoss()

    input = Variable(torch.randn(4, 20, 10, 10)).cuda() # batch, input_channels, height, width
    target = Variable(torch.randn(4, 20, 10, 10)).double().cuda() # batch, output_channels, height, width

    output = convlstm(input)
    output = output[0][0].double()
    res = torch.autograd.gradcheck(loss_fn, (output, target[:,0,:,:,:,:]), eps=1e-6, raise_exception=True)
    print(res)
