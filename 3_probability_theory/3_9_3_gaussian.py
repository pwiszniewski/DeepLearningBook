import matplotlib
matplotlib.use('Qt5Agg')


import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Slider, Button, RadioButtons

mi, std = 0, 1

def get_gauss(x, mi, var):
    return np.sqrt(1/(2*np.pi*var))*np.exp(-1/(2*var)*(x-mi)**2)


def redraw(ax):
    ax.clear()
    x = np.linspace(*xlim, num=100)
    y = get_gauss(x, mi, std**2)
    ax.scatter(x, y, s=1)
    ax.set_ylim(0, 1)
    fig.canvas.draw_idle()

xlim = -5, 5
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.2, bottom=0.25)
x = np.linspace(*xlim, num=100)
y = get_gauss(x, mi, std**2)
ax.scatter(x, y, s=1)
# ax.set_xlim(-5, 5)
ax.set_ylim(0, 1)


axcolor = 'lightgoldenrodyellow'
axa = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
axb = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
# axc = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
# axd = plt.axes([0.25, 0.0, 0.65, 0.03], facecolor=axcolor)

sa = Slider(axa, 'mi', -10, 10, valinit=mi, valstep=.1)
sb = Slider(axb, 'std', 0, 10, valinit=std, valstep=.1)
# sc = Slider(axc, 'c', -10, 10, valinit=c, valstep=.1)
# sd = Slider(axd, 'd', -10, 10, valinit=d, valstep=.1)


def update_a(val):
    global mi
    mi = val
    redraw(ax)

def update_b(val):
    global std
    std = val
    redraw(ax) 

sa.on_changed(update_a)
sb.on_changed(update_b)
# sc.on_changed(update_c)
# sd.on_changed(update_d)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    sa.reset()
    sb.reset()
    # sc.reset()
    # sd.reset()
button.on_clicked(reset)


plt.show()
