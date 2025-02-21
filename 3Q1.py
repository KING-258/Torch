import torch
import matplotlib.pyplot as plt
x = torch.tensor( [12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4, 19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2])
y = torch.tensor( [11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6, 16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6])
a = torch.tensor(0.001)
n = len(x)
w, b = torch.tensor(1.0, requires_grad= True), torch.tensor(1.0, requires_grad= True)
losses = []
epochs = 10
for epoch in range(epochs):
    loss = 0.0
    for i in range(n):
        y_pred = (w * x[i]) + b
        loss += (y[i] - y_pred) ** 2
    loss /= n
    loss.backward()
    with torch.no_grad():
        w -= a * w.grad
        b -= a * b.grad
    w.grad.zero_()
    b.grad.zero_()
    if(epoch % 25 == 0):
        print(f'parameters = w = {w.item()}, b = {b.item()}, loss = {loss.item()}')
    losses.append(loss.detach().numpy())
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
plt.plot(losses)
plt.show()