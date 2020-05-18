# The mechanics of learning

Training a neural network essentially involves changing the model for a slightly more elaborate one with a few (or a metric ton) more parameters

Compute  theindividual derivatives of the loss with respect to each parameter and put them in a vec-tor of derivatives: the gradient.

A training iteration during which you update the parameters for all your training samples is called an `epoch`.

Applying a change to w that’s proportional to the rate of change of the loss is a good idea,  especially when the loss has several parameters. Your training process blew up, leading to losses becoming inf. This result is a clear sign that params is receiving updates that are too large. Diverging optimization on convex function (parabolalike) due to large steps.  

**How to limit loss from blowing up?** 
- You could simply choose a smaller `learning_rate`. You usually change learning rates by order of magnitude.  But there’s another problem: updates to parametersare small, so the loss decreases slowly and eventually stalls. You could obviate this issueby making the learning_rate adaptive.
- Change the inputs so that the gradients aren’t so different. You can make sure that the range of the input doesn’t get too far from **the range of -1.0 to 1.0**, roughly speaking

**Demo of Gradient Descent**
```python
# t_u means the unknown temperature data
# t_c means the ground truth temperature data
# t_c means the predicted temperature data

# model definition
def model(t_u,w,b):
    return w * t_u + b

# loss function definition
def loss_fn(t_p,t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

# gradient of loss function
def dloss_fn(t_p,t_c):
    squared_diffs = (t_p - t_c) * 2
    return squared_diffs

# gradient of model with respect to w
def dmodel_dw(t_u,w,b):
    return t_u

# gradient of model with respect to b
def dmodel_db(t_u,w,b):
    return 1.0

# gradient descent
def grad_fn(t_u,t_c,t_p,w,b):
    dloss_dw = dloss_fn(t_p,t_c) * dmodel_dw(t_u,w,b)
    dloss_db = dloss_fn(t_p,t_c) * dmodel_db(t_u,w,b)
    return torch.stack([dloss_dw.mean(),dloss_db.mean()])

# epoch of training
def training_loop(n_epochs,learning_rate,params,t_u,t_c,print_detail=True):
    for epoch in range(1,n_epochs+1):
        w,b = params
        t_p = model(t_u,w,b)
        loss = loss_fn(t_p,t_c)
        grad = grad_fn(t_u,t_c,t_p,w,b)
        params = params - learning_rate * grad
        if print_detail:
            print('Epoch %d, Loss %f' % (epoch,float(loss)))
            print(f'\tParams : {params}')
            print(f'\tgrad : {grad}')
    return params

# begin training
params = training_loop(
    n_epochs = 5000,
    learning_rate = 1e-2,
    params = torch.tensor([1.0,0.0]),
    t_u = t_un,
    t_c = t_c,
    print_detail = False
)
```
```python
%matplotlib inline
from matplotlib import pyplot as plt

# Plot the linear model
t_p = model(t_un,*params)
print(f"t_p : {t_p}")
fig = plt.figure(dpi=600)
plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
plt.plot(t_u.numpy(),t_p.detach().numpy())
plt.plot(t_u.numpy(),t_c.numpy(),'o')
```

## Demo of Stochastic Gradient Descend
```python
step_size = 0.01
linear_module = nn.Linear(d,1)
loss_func = nn.MSELoss()
optim = torch.optim.SGD(linear_module.parameters(),lr=step_size)
print("iter,\tloss,\tw")
for i in range(200):
    rand_idx = np.random.choice(n)
    x = X[rand_idx]
    y_hat = linear_module(x)
    loss = loss_func(y_hat,y[rand_idx])
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    if i % 20 == 0:
        print("{},\t{:.2f},\t{}"
              .format(i,loss.item(),linear_module.weight
                      .view(2).detach().numpy()))
        
print("\ntrue w:\t\t",true_w.view(2).numpy())
print("estimated w:\t",linear_module.weight.view(2).detach().numpy())
```