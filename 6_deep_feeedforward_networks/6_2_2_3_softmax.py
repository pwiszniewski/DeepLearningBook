import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def f(t, amplitude, frequency):
    return amplitude * np.sin(2 * np.pi * frequency * t)

x = [0, 1, 2]

init_amplitude = 1

fig, ax = plt.subplots()
# line, = plt.plot(t, f(t, init_amplitude, init_frequency), lw=2)
ax.bar(x, [1/3, 1/3, 1/3])
ax.set_ylim(0, 1)
ax.set_xlabel('Time [s]')

plt.subplots_adjust(left=0.25, bottom=0.25)

ax1 = plt.axes([0.05, 0.25, 0.0225, 0.63])
slider1 = Slider(
    ax=ax1,
    label="0",
    valmin=-10,
    valmax=10,
    valinit=init_amplitude,
    orientation="vertical"
)

ax2 = plt.axes([0.1, 0.25, 0.0225, 0.63])
slider2 = Slider(
    ax=ax2,
    label="1",
    valmin=-10,
    valmax=10,
    valinit=init_amplitude,
    orientation="vertical"
)

ax3 = plt.axes([0.15, 0.25, 0.0225, 0.63])
slider3 = Slider(
    ax=ax3,
    label="2",
    valmin=-10,
    valmax=10,
    valinit=init_amplitude,
    orientation="vertical"
)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def update(val):
    ax.clear()
    ax.bar([0, 1, 2], softmax([slider1.val, slider2.val, slider3.val]))
    ax.set_ylim(0, 1)
    fig.canvas.draw_idle()


slider1.on_changed(update)
slider2.on_changed(update)
slider3.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    slider1.reset()
    slider2.reset()
    slider3.reset()
button.on_clicked(reset)

plt.show()