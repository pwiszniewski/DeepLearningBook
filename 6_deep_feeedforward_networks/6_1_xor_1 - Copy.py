import matplotlib
matplotlib.use('Qt5Agg')


import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Slider, Button, RadioButtons

a,b,c,d = 1,0,0,1   # coeffs of hidden layer matrix
e,f = 0,0           # constants of hidden layer
g, h = 0,0           # coeffs of final layer
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# X = np.array([[1,1], [0,0], [-1,-1], [1,-1], [-1,1], [0,1], [1,0], [-1,0], [0,-1]])
# y = np.array(range(len(X)))

def ReLU(x):
    return np.maximum(x, 0)

def draw_path(ax):
    global a, b, c, d
    W = np.array([[a, b], 
              [c, d]])
    C = np.array([e, f])          
    X_trans = ReLU(X @ W + C)
    ax.scatter(X_trans[:,0], X_trans[:,1], c=y)
    

    
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.2, bottom=0.25)
ax.scatter(X[:,0], X[:,1], c=y)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect(1)


axcolor = 'lightgoldenrodyellow'
axa = plt.axes([0.25, 0.15, 0.3, 0.03], facecolor=axcolor)
axb = plt.axes([0.25, 0.1, 0.3, 0.03], facecolor=axcolor)
axc = plt.axes([0.25, 0.05, 0.3, 0.03], facecolor=axcolor)
axd = plt.axes([0.25, 0.0, 0.3, 0.03], facecolor=axcolor)
axe = plt.axes([0.6, 0.15, 0.3, 0.03], facecolor=axcolor)
axf = plt.axes([0.6, 0.1, 0.3, 0.03], facecolor=axcolor)

sa = Slider(axa, 'a', -10, 10, valinit=a, valstep=.1)
sb = Slider(axb, 'b', -10, 10, valinit=b, valstep=.1)
sc = Slider(axc, 'c', -10, 10, valinit=c, valstep=.1)
sd = Slider(axd, 'd', -10, 10, valinit=d, valstep=.1)
se = Slider(axe, 'e', -10, 10, valinit=d, valstep=.1)
sf = Slider(axf, 'f', -10, 10, valinit=d, valstep=.1)


def update_a(val):
    global a
    ax.clear()
    a = val
    draw_path(ax)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()

def update_b(val):
    global b
    ax.clear()
    b = val
    draw_path(ax)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()

def update_c(val):
    global c
    ax.clear()
    c = val
    draw_path(ax)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()

def update_d(val):
    global d
    ax.clear()
    d = val
    draw_path(ax)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()

def update_e(val):
    global e
    ax.clear()
    e = val
    draw_path(ax)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()

def update_f(val):
    global f
    ax.clear()
    f = val
    draw_path(ax)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()

def update_g(val):
    global g
    ax.clear()
    g = val
    draw_path(ax)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()

def update_h(val):
    global h
    ax.clear()
    h = val
    draw_path(ax)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.canvas.draw_idle()


sa.on_changed(update_a)
sb.on_changed(update_b)
sc.on_changed(update_c)
sd.on_changed(update_d)
se.on_changed(update_e)
sf.on_changed(update_f)
sf.on_changed(update_g)
sf.on_changed(update_h)

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
