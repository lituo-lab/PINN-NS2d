import torch
import matplotlib.pyplot as plt
from net import MLP

model = MLP()
model.load_state_dict(torch.load('model.param'))

t0 = 0.6
xc = torch.linspace(0, 6.28, 100)
yc = torch.linspace(0, 6.28, 100)
xx, yy = torch.meshgrid(xc, yc)
xx = xx.reshape(-1, 1)
yy = yy.reshape(-1, 1)
tt = torch.zeros_like(xx) + t0

u_actu = -torch.cos(xx)*torch.sin(yy)*torch.exp(-2*tt)
v_actu =  torch.sin(xx)*torch.cos(yy)*torch.exp(-2*tt)
p_actu =  -0.25 *( torch.cos(2*xx)+torch.cos(2*yy)) *torch.exp(-4*tt)

u_pred, v_pred, p_pred = model(torch.cat([xx, yy, tt], dim=1))

u_pred = u_pred.reshape(100, 100).detach().numpy()
u_actu = u_actu.reshape(100, 100).detach().numpy()

f = plt.figure()
plt.subplot(121)    
plt.imshow(u_actu, origin='lower')
plt.title(f'u_actu t={t0}s')
plt.subplot(122)
plt.imshow(u_pred, origin='lower')
plt.title(f'u_pred t={t0}s')
plt.show()


v_pred = v_pred.reshape(100, 100).detach().numpy()
v_actu = v_actu.reshape(100, 100).detach().numpy()

f = plt.figure()
plt.subplot(121)       
plt.imshow(v_actu, origin='lower')
plt.title(f'v_actu t={t0}s')
plt.subplot(122)
plt.imshow(v_pred, origin='lower')
plt.title(f'v_pred t={t0}s')
plt.show()

p_pred = p_pred.reshape(100, 100).detach().numpy()
p_actu = p_actu.reshape(100, 100).detach().numpy()
f = plt.figure()
plt.subplot(121)       
plt.imshow(p_actu, origin='lower')
plt.title(f'p_actu t={t0}s')
plt.subplot(122)
plt.imshow(p_pred, origin='lower')
plt.title(f'p_pred t={t0}s')
plt.show()







