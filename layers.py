import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

class GraphConvolution(Module):
    """
    GCN layer, based on https://arxiv.org/abs/1609.02907
    that allows MIMO
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu, ):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.a = Parameter(torch.empty(size=(2 * out_features, 1)))
        self.dropout = dropout
        self.act = act
        # self.leakyrelu = nn.LeakyReLU()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def causality_message(self):
        pass

    def _prepare_causality_attention_input(self, Wh):

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.act(e)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        """
        if the input features are a matrix -- excute regular GCN,
        if the input features are of shape [K, N, Z] -- excute MIMO GCN with shared weights.
        """

       # naive gcn message-passing
        support = torch.stack(
                [torch.mm(inp, self.weight) for inp in torch.unbind(input, dim=0)],
                dim=0)

        output = torch.stack(
                [torch.spmm(adj, sup) for sup in torch.unbind(support, dim=0)],
                dim=0)

        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Causal_GraphConvolution(Module):
    """
    GCN layer, based on https://arxiv.org/abs/1609.02907
    that allows MIMO
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu, ):
        super(Causal_GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.a = Parameter(torch.empty(size=(2 * out_features, 1)))
        self.dropout = dropout
        self.act = act
        # self.leakyrelu = nn.LeakyReLU()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def causality_message(self):
        pass

    def _prepare_causality_attention_input(self, Wh):

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.act(e)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        """
        if the input features are a matrix -- excute regular GCN,
        if the input features are of shape [K, N, Z] -- excute MIMO GCN with shared weights.
        """

        # causality-aware message calculation
        Whs = []
        for inp in torch.unbind(input, dim=0):
            Wh = torch.mm(inp, self.weight) # torch.Size([32, 16])
            e = self._prepare_causality_attention_input(Wh)

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, Wh)

            Whs.append(h_prime)

        support = torch.stack(Whs,dim=0) #torch.Size([35, 32, 16])


        output = torch.stack(
                [torch.spmm(adj, sup) for sup in torch.unbind(support, dim=0)],
                dim=0)

        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
