#%% import
import os
import torch
from net import MLP
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% auto gradient

def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


#%% plot_predict_now

def plot_predict_now(epoch, model):
    
    with torch.no_grad():
        t0 = 0.5
        xc = torch.linspace(0, 6.28, 100)
        yc = torch.linspace(0, 6.28, 100)
        xx, yy = torch.meshgrid(xc, yc)
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        tt = torch.zeros_like(xx) + t0
    
        p_actu = -0.25*(torch.cos(2*xx)+torch.cos(2*yy))*torch.exp(-4*tt)
    
        model.eval()
        
        in_xyt = torch.cat([xx, yy, tt], dim=1).to(DEVICE)
        
        u_pred, v_pred, p_pred = model(in_xyt)
        
        p_pred = p_pred.to('cpu')
        p_pred = p_pred.reshape(100, 100).detach().numpy()
        p_actu = p_actu.reshape(100, 100).detach().numpy()


    path = './output/point128'
    if not os.path.exists(path):
        os.makedirs(path)

    plt.figure()
    plt.suptitle(f'epoch={epoch}, cross profile(t=0.5)')
    plt.subplot(121)
    plt.imshow(p_actu, origin='lower')
    plt.title('p_actu')
    plt.subplot(122)
    plt.imshow(p_pred, origin='lower')
    plt.title('p_pred')
    
    plt.savefig(path + f'/epoch-{epoch}.png')
    plt.close('all')


#%% Loss

def loss_nsEqu(model):

    # Neqns
    n = 20000
    
    torch.manual_seed(2023)
    x = 2*torch.pi*(torch.rand(n, 1)).requires_grad_(True).to(DEVICE)
    y = 2*torch.pi*(torch.rand(n, 1)).requires_grad_(True).to(DEVICE)
    t = 2*(torch.rand(n, 1)).requires_grad_(True).to(DEVICE)

    u, v, p = model(torch.cat([x, y, t], dim=1))
    px = gradients(p, x, 1)
    py = gradients(p, y, 1)

    ut = gradients(u, t, 1)
    ux = gradients(u, x, 1)
    uy = gradients(u, y, 1)
    uxx = gradients(u, x, 2)
    uyy = gradients(u, y, 2)

    vt = gradients(v, t, 1)
    vx = gradients(v, x, 1)
    vy = gradients(v, y, 1)
    vxx = gradients(v, x, 2)
    vyy = gradients(v, y, 2)

    loss_fn = torch.nn.MSELoss()
    e1 = loss_fn((ut + u*ux + v*uy), -px + (uxx + uyy))
    e2 = loss_fn((vt + u*vx + v*vy), -py + (vxx + vyy))
    e4 = loss_fn(ux, -vy)

    return e1 + e2 + e4


def loss_Ndata(model, Ndata):
    
    xi, yi, ti, ui, vi = Ndata
    u, v, p = model(torch.cat([xi, yi, ti], dim=1))
    loss_fn = torch.nn.MSELoss()

    return loss_fn(u, ui) + loss_fn(v, vi)


#%% Ndata
N = 128
torch.manual_seed(1)
xi = 2*torch.pi*torch.rand(N, 1).to(DEVICE)
yi = 2*torch.pi*torch.rand(N, 1).to(DEVICE)
ti = 2*torch.rand(N, 1).to(DEVICE)
ui = -torch.cos(xi)*torch.sin(yi)*torch.exp(-2*ti)
vi = torch.sin(xi)*torch.cos(yi)*torch.exp(-2*ti)

Ndata = [xi, yi, ti, ui, vi]

# %% Training
model = MLP().to(DEVICE)
# model.load_state_dict(torch.load('model.param'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 20000

for epoch in range(epochs):
    
    model.train()
    optimizer.zero_grad()
    loss1 = loss_nsEqu(model)
    loss2 = loss_Ndata(model, Ndata)
    loss = loss2 + loss1
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:  
        epoch_epoch =  f'epoch={epoch}/{epochs}'
        epoch_loss1 = 'loss1='+ str(format(loss1.item(), '.8f'))
        epoch_loss2 = 'loss2='+ str(format(loss2.item(), '.8f'))
        print(epoch_epoch, epoch_loss1, epoch_loss2)

    if epoch % 200 == 0:
        torch.save(model.state_dict(), 'model.param')
        plot_predict_now(epoch, model)


