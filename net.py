
import torch

class Swish(torch.nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x * torch.sigmoid(10*self.beta * x)
    
    
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net_u = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            Swish(),
            torch.nn.Linear(64, 64),
            Swish(),
            torch.nn.Linear(64, 64),
            Swish(),
            torch.nn.Linear(64, 64),
            Swish(),
            torch.nn.Linear(64, 64),
            Swish(),
            torch.nn.Linear(64, 64),
            Swish(),
            torch.nn.Linear(64, 64),
            Swish(),
            torch.nn.Linear(64, 1)
        )

        self.net_v = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            Swish(),
            torch.nn.Linear(64, 64),
            Swish(),
            torch.nn.Linear(64, 64),
            Swish(),
            torch.nn.Linear(64, 64),
            Swish(),
            torch.nn.Linear(64, 64),
            Swish(),
            torch.nn.Linear(64, 64),
            Swish(),
            torch.nn.Linear(64, 64),
            Swish(),
            torch.nn.Linear(64, 1)
        )
        
        self.net_p = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            Swish(),
            torch.nn.Linear(64, 64),
            Swish(),
            torch.nn.Linear(64, 64),
            Swish(),
            torch.nn.Linear(64, 64),
            Swish(),
            torch.nn.Linear(64, 64),
            Swish(),
            torch.nn.Linear(64, 64),
            Swish(),
            torch.nn.Linear(64, 64),
            Swish(),
            torch.nn.Linear(64, 1)
        )


    def forward(self, x):
        return self.net_u(x), self.net_v(x), self.net_p(x)

if __name__ == '__main__':
    
    model = MLP()
    
    in_xyt = torch.randn(10,3)
    
    u_pred, v_pred, p_pred = model(in_xyt)
    
    print(in_xyt.shape)
    print(u_pred.shape)
    print(v_pred.shape)
    print(p_pred.shape)


