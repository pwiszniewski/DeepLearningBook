import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

N = 9
x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, N)
# full coordinate arrays
xx, yy = np.meshgrid(x, y)

fig, ax = plt.subplots(1, 1)



ax.scatter(xx, yy, s=1)
UV = []
colors = []
for i in range(N):
    for j in range(N):
        UV.append(softmax(np.array([xx[i, j], yy[i, j]])))
        c = 0 if xx[i, j] == yy[i, j] else 1 if xx[i, j] > yy[i, j] else -1
        colors.append(c)
UV = list(zip(*UV))
UV = np.array(UV).reshape(2, N, N)
colors = np.array(colors).reshape(N, N)
U = UV[0] - xx
V = UV[1] - yy

ax.quiver(xx, yy, U, V, colors, angles='xy', scale_units='xy', scale=1, width=.003)
rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
ax.scatter(0, 0, color='red')
ax.set_aspect('equal')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
plt.grid(b=True)


ax1 = plt.axes([0.05, 0.25, 0.0225, 0.63])
slider1 = Slider(
    ax=ax1,
    label="x_shift",
    valmin=-10,
    valmax=10,
    valinit=0,
    orientation="vertical"
)

ax2 = plt.axes([0.1, 0.25, 0.0225, 0.63])
slider2 = Slider(
    ax=ax2,
    label="y_shift",
    valmin=-10,
    valmax=10,
    valinit=0,
    orientation="vertical"
)

ax3 = plt.axes([0.15, 0.25, 0.0225, 0.63])
slider3 = Slider(
    ax=ax3,
    label="xy_shift",
    valmin=-10,
    valmax=10,
    valinit=0,
    orientation="vertical"
)


x_shift, y_shift = 0, 0

def update_plot():
    ax.clear()
    xx_shifted = xx + x_shift
    yy_shifted = yy + y_shift
    ax.scatter(xx_shifted, yy_shifted, s=1)
    UV = []
    colors = []
    for i in range(N):
        for j in range(N):
            UV.append(softmax(np.array([xx_shifted[i, j], yy_shifted[i, j]])))
            c = 0 if xx_shifted[i, j] == yy_shifted[i, j] else 1 if xx_shifted[i, j] > yy_shifted[i, j] else -1
            colors.append(c)
    UV = list(zip(*UV))
    UV = np.array(UV).reshape(2, N, N)
    colors = np.array(colors).reshape(N, N)
    U = UV[0] - xx_shifted
    V = UV[1] - yy_shifted

    ax.quiver(xx_shifted, yy_shifted, U, V, colors, angles='xy', scale_units='xy', scale=1, width=.003)
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.scatter(0, 0, color='red')
    ax.set_aspect('equal')
    ax.grid(b=True)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    fig.canvas.draw_idle()

def update_xshift(val):
    global x_shift
    x_shift = val
    update_plot()

def update_yshift(val):
    global y_shift
    y_shift = val
    update_plot()

def update_xyshift(val):
    global x_shift
    global y_shift
    x_shift = val
    y_shift = val
    update_plot()

slider1.on_changed(update_xshift)
slider2.on_changed(update_yshift)
slider3.on_changed(update_xyshift)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    slider1.reset()
    slider2.reset()
    slider3.reset()
button.on_clicked(reset)

plt.show()