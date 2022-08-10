import matplotlib
matplotlib.use('Qt5Agg')
# solution: W = [[1 1] [1 1]] c = [0 −1] w = [1 -2] b = 0 (always in this case)
# another solution: W = [[1 -1] [-1 1]] c = [0 0] w = [1 1] b = 0 (always in this case)

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Slider, Button, RadioButtons

a,b,c,d = 1,0,0,1   # coeffs of hidden layer matrix
e,f = 0,0           # constants of hidden layer
g, h = 0,0           # coeffs of final layer
activation1 = 'ReLU'
activation2 = 'None'
out_type = 'binary'
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# X = np.array([[1,1], [0,0], [-1,-1], [1,-1], [-1,1], [0,1], [1,0], [-1,0], [0,-1]])
# y = np.array(range(len(X)))

x_grid = np.arange(-3.0,3.0,0.1)
y_grid = np.arange(-3.0,3.0,0.1)

def ReLU(x):
    return np.maximum(x, 0)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def draw_original(ax):
    global a, b, c, d
    W = np.array([[a, b], 
              [c, d]])
    C = np.array([e, f])  

    # plot decision boundary
    X_grid,Y_grid = np.meshgrid(x_grid, y_grid) # grid of point    
    XX = np.concatenate([X_grid.reshape(1,-1), Y_grid.reshape(1,-1)], axis=0).T
    X_trans = XX @ W + C
    if activation1 == 'ReLU':
        X_trans = ReLU(X_trans)
    w_out = np.array([g, h]).reshape(2, 1)
    out = X_trans @ w_out
    if activation2 == 'sigmoid':
        out = sigmoid(out)
    colors = out > .5 if out_type == 'binary' else out
    ax.scatter(XX[:,0], XX[:,1], c=colors, cmap=plt.cm.Reds)

    X_trans = X @ W + C
    if activation1 == 'ReLU':
        X_trans = ReLU(X_trans)
    ax.scatter(X[:,0], X[:,1], c=y)

def draw_transformed(ax):
    global a, b, c, d
    W = np.array([[a, b], 
              [c, d]])
    C = np.array([e, f])  

    # plot decision boundary
    X_grid,Y_grid = np.meshgrid(x_grid, y_grid) # grid of point    
    XX = np.concatenate([X_grid.reshape(1,-1), Y_grid.reshape(1,-1)], axis=0).T
    # print(np.concatenate([np.ravel(X_grid), np.ravel(Y_grid)]))
    X_trans = XX @ W + C
    if activation1 == 'ReLU':
        X_trans = ReLU(X_trans)
    w_out = np.array([g, h]).reshape(2, 1)
    out = X_trans @ w_out
    if activation2 == 'sigmoid':
        out = sigmoid(out)
    colors = out > .5 if out_type == 'binary' else out
    ax.scatter(X_trans[:,0], X_trans[:,1], c=colors, cmap=plt.cm.Reds)

    X_trans = X @ W + C
    if activation1 == 'ReLU':
        X_trans = ReLU(X_trans)
    ax.scatter(X_trans[:,0], X_trans[:,1], c=y)
    #plt.contourf(x,y,z)
    


    
fig, ax = plt.subplots(1, 2)
plt.subplots_adjust(left=0.2, bottom=0.25)
ax0 = ax[0]
ax = ax[1]

ax.scatter(X[:,0], X[:,1], c=y)
ax.plot(X[:,0], X[:,1], c=y)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect(1)

ax0.scatter(X[:,0], X[:,1], c=y)
ax0.plot(X[:,0], X[:,1], c=y)
ax0.set_xlim(-5, 5)
ax0.set_ylim(-5, 5)
ax0.set_aspect(1)



axcolor = 'lightgoldenrodyellow'
rax1 = plt.axes([0.05, 0.7, 0.15, 0.15], facecolor=axcolor)
rax2 = plt.axes([0.05, 0.5, 0.15, 0.15], facecolor=axcolor)
rax3 = plt.axes([0.05, 0.3, 0.15, 0.15], facecolor=axcolor)
radio1 = RadioButtons(rax1, ('ReLU', 'None'))
radio2 = RadioButtons(rax2, ('sigmoid', 'None'), active=1)
radio3 = RadioButtons(rax3, ('binary', 'raw'))

axa = plt.axes([0.25, 0.15, 0.3, 0.03], facecolor=axcolor)
axb = plt.axes([0.25, 0.1, 0.3, 0.03], facecolor=axcolor)
axc = plt.axes([0.25, 0.05, 0.3, 0.03], facecolor=axcolor)
axd = plt.axes([0.25, 0.0, 0.3, 0.03], facecolor=axcolor)
axe = plt.axes([0.6, 0.15, 0.3, 0.03], facecolor=axcolor)
axf = plt.axes([0.6, 0.1, 0.3, 0.03], facecolor=axcolor)
axg = plt.axes([0.6, 0.05, 0.3, 0.03], facecolor=axcolor)
axh = plt.axes([0.6, 0.0, 0.3, 0.03], facecolor=axcolor)

sa = Slider(axa, 'a', -10, 10, valinit=a, valstep=.1)
sb = Slider(axb, 'b', -10, 10, valinit=b, valstep=.1)
sc = Slider(axc, 'c', -10, 10, valinit=c, valstep=.1)
sd = Slider(axd, 'd', -10, 10, valinit=d, valstep=.1)
se = Slider(axe, 'e', -10, 10, valinit=e, valstep=.1)
sf = Slider(axf, 'f', -10, 10, valinit=f, valstep=.1)
sg = Slider(axg, 'g', -10, 10, valinit=g, valstep=.1)
sh = Slider(axh, 'h', -10, 10, valinit=h, valstep=.1)


def update_a(val):
    global a
    a = val
    ax0.clear()
    ax.clear()    
    draw_original(ax0)
    draw_transformed(ax)
    ax0.set_xlim(-5, 5)
    ax0.set_ylim(-5, 5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()

def update_b(val):
    global b
    b = val
    ax0.clear()
    ax.clear()    
    draw_original(ax0)
    draw_transformed(ax)
    ax0.set_xlim(-5, 5)
    ax0.set_ylim(-5, 5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()

def update_c(val):
    global c
    c = val
    ax0.clear()
    ax.clear()    
    draw_original(ax0)
    draw_transformed(ax)
    ax0.set_xlim(-5, 5)
    ax0.set_ylim(-5, 5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()

def update_d(val):
    global d
    d = val
    ax0.clear()
    ax.clear()    
    draw_original(ax0)
    draw_transformed(ax)
    ax0.set_xlim(-5, 5)
    ax0.set_ylim(-5, 5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()

def update_e(val):
    global e
    e = val
    ax0.clear()
    ax.clear()    
    draw_original(ax0)
    draw_transformed(ax)
    ax0.set_xlim(-5, 5)
    ax0.set_ylim(-5, 5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()

def update_f(val):
    global f
    f = val
    ax0.clear()
    ax.clear()    
    draw_original(ax0)
    draw_transformed(ax)
    ax0.set_xlim(-5, 5)
    ax0.set_ylim(-5, 5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()

def update_g(val):
    global g
    g = val
    ax0.clear()
    ax.clear()    
    draw_original(ax0)
    draw_transformed(ax)
    ax0.set_xlim(-5, 5)
    ax0.set_ylim(-5, 5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()

def update_h(val):
    global h
    h = val
    ax0.clear()
    ax.clear()    
    draw_original(ax0)
    draw_transformed(ax)
    ax0.set_xlim(-5, 5)
    ax0.set_ylim(-5, 5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()

def switch_activation1(label):
    global activation1
    activation1 = label
    ax0.clear()
    ax.clear()    
    draw_original(ax0)
    draw_transformed(ax)
    ax0.set_xlim(-5, 5)
    ax0.set_ylim(-5, 5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()

def switch_activation2(label):
    global activation2
    activation2 = label
    ax0.clear()
    ax.clear()    
    draw_original(ax0)
    draw_transformed(ax)
    ax0.set_xlim(-5, 5)
    ax0.set_ylim(-5, 5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()

def switch_out_type(label):
    global out_type
    out_type = label
    ax0.clear()
    ax.clear()    
    draw_original(ax0)
    draw_transformed(ax)
    ax0.set_xlim(-5, 5)
    ax0.set_ylim(-5, 5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()


sa.on_changed(update_a)
sb.on_changed(update_b)
sc.on_changed(update_c)
sd.on_changed(update_d)
se.on_changed(update_e)
sf.on_changed(update_f)
sg.on_changed(update_g)
sh.on_changed(update_h)
radio1.on_clicked(switch_activation1)
radio2.on_clicked(switch_activation2)
radio3.on_clicked(switch_out_type)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    sa.reset()
    sb.reset()
    sc.reset()
    sd.reset()
    se.reset()
    sf.reset()
    sg.reset()
    sh.reset()
button.on_clicked(reset)


plt.show()
