import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

init_z0 = 0

fig, ax = plt.subplots(nrows=2)
z1 = np.linspace(-10, 10, 100)
y0 = [softmax([init_z0, zz])[0] for zz in z1]
y1 = [softmax([init_z0, zz])[1] for zz in z1]
line0, = ax[0].plot(z1, y0, lw=2)
line1, = ax[1].plot(z1, y1, lw=2)

plt.subplots_adjust(left=0.25, bottom=0.25)

ax1 = plt.axes([0.05, 0.25, 0.0225, 0.63])
slider1 = Slider(
    ax=ax1,
    label="z0",
    valmin=-10,
    valmax=10,
    valinit=init_z0,
    orientation="vertical"
)



def update(val):
    y0 = [softmax([slider1.val, zz])[0] for zz in z1]
    y1 = [softmax([slider1.val, zz])[1] for zz in z1]
    line0.set_ydata(y0)
    line1.set_ydata(y1)
    fig.canvas.draw_idle()


slider1.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    slider1.reset()
button.on_clicked(reset)

plt.show()