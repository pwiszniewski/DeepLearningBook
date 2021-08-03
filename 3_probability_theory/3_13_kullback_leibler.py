import matplotlib
matplotlib.use('Qt5Agg')


import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea

mip, stdp = -4, 1
mip2, stdp2 = 4, 1
miq, stdq = 0, 1


xlim = -10, 10
x = np.linspace(*xlim, num=500)

def get_gauss(x, mi, var):
    return np.sqrt(1/(2*np.pi*var))*np.exp(-1/(2*var)*(x-mi)**2)

def get_kl_divergence(p, q):
    return np.sum(p * np.log2(p/q))

# def redraw(ax):
#     ax.clear()
#     yp = .5*get_gauss(x, mip, stdp**2) + .5*get_gauss(x, mip2, stdp2**2)
#     ax.scatter(x, yp, s=1)
#     yq = get_gauss(x, miq, stdq**2)
#     ax.scatter(x, yq, s=1)
#     kl_pq = get_kl_divergence(yp, yq)
#     if kl_pq < 1:
#         kl_pqstr = f'{kl_pq:.1f}'
#     else:
#         kl_pqstr = f'{kl_pq:.0f}'
#     kl_qp = get_kl_divergence(yq, yp)
#     if kl_qp < 1:
#         kl_qpstr = f'{kl_qp:.1f}'
#     else:
#         kl_qpstr = f'{kl_qp:.0f}'
#     ax.annotate(f'kl_pq: {kl_pqstr}, kl_qp: {kl_qpstr}', xy=(0, 1), xycoords="axes fraction",
#             va="top", ha="left",
#             bbox=dict(boxstyle="round", fc="w"), fontsize=20)
#     # ax.set_xlim(-5, 5)
#     ax.set_ylim(0, 1)
#     ax.legend(['p(x)', 'q*(x)'])
#     fig.canvas.draw_idle()


def redraw(scp, scq, ann):
    # ax.clear()
    yp = .5*get_gauss(x, mip, stdp**2) + .5*get_gauss(x, mip2, stdp2**2)
    scp.set_offsets(np.c_[x,yp])
    # ax.scatter(x, yp, s=1)
    yq = get_gauss(x, miq, stdq**2)
    scq.set_offsets(np.c_[x,yq])
    # ax.scatter(x, yq, s=1)
    kl_pq = get_kl_divergence(yp, yq)
    if kl_pq < 1:
        kl_pqstr = f'{kl_pq:.1f}'
    else:
        kl_pqstr = f'{kl_pq:.0f}'
    kl_qp = get_kl_divergence(yq, yp)
    if kl_qp < 1:
        kl_qpstr = f'{kl_qp:.1f}'
    else:
        kl_qpstr = f'{kl_qp:.0f}'
    ann.set_text(f'kl_pq: {kl_pqstr}, kl_qp: {kl_qpstr}')
    # ax.annotate(f'kl_pq: {kl_pqstr}, kl_qp: {kl_qpstr}', xy=(0, 1), xycoords="axes fraction",
    #         va="top", ha="left",
    #         bbox=dict(boxstyle="round", fc="w"), fontsize=20)
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(0, 1)
    # ax.legend(['p(x)', 'q*(x)'])
    fig.canvas.draw_idle()




# init plot
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.2, bottom=0.25)


yp = .5*get_gauss(x, mip, stdp**2) + .5*get_gauss(x, mip2, stdp2**2)
scp = ax.scatter(x, yp, s=1)
yq = get_gauss(x, miq, stdq**2)
scq = ax.scatter(x, yq, s=1)
kl_pq = get_kl_divergence(yp, yq)
if kl_pq < 1:
    kl_pqstr = f'{kl_pq:.1f}'
else:
    kl_pqstr = f'{kl_pq:.0f}'
kl_qp = get_kl_divergence(yq, yp)
if kl_qp < 1:
    kl_qpstr = f'{kl_qp:.1f}'
else:
    kl_qpstr = f'{kl_qp:.0f}'
ann = ax.annotate(f'kl_pq: {kl_pqstr}, kl_qp: {kl_qpstr}', xy=(0, 1), xycoords="axes fraction",
        va="top", ha="left",
        bbox=dict(boxstyle="round", fc="w"), fontsize=20)
# ax.set_xlim(-5, 5)
ax.set_ylim(0, 1)
ax.legend(['p(x)', 'q*(x)'])


axcolor = 'lightgoldenrodyellow'
axa = plt.axes([0.25, 0.15, 0.3, 0.03], facecolor=axcolor)
axb = plt.axes([0.25, 0.1, 0.3, 0.03], facecolor=axcolor)
axc = plt.axes([0.25, 0.05, 0.3, 0.03], facecolor=axcolor)
axd = plt.axes([0.25, 0.0, 0.3, 0.03], facecolor=axcolor)
axe = plt.axes([0.6, 0.15, 0.3, 0.03], facecolor=axcolor)
axf = plt.axes([0.6, 0.1, 0.3, 0.03], facecolor=axcolor)

sa = Slider(axa, 'mip1', -10, 10, valinit=mip, valstep=.1)
sb = Slider(axb, 'stdp1', 0, 10, valinit=stdp, valstep=.1)
sc = Slider(axc, 'mip2', -10, 10, valinit=mip2, valstep=.1)
sd = Slider(axd, 'stdp2', 0, 10, valinit=stdp2, valstep=.1)
se = Slider(axe, 'miq', -10, 10, valinit=miq, valstep=.1)
sf = Slider(axf, 'stdq', 0, 10, valinit=stdp2, valstep=.1)


def update_a(val):
    global mip
    mip = val
    redraw(scp, scq, ann)

def update_b(val):
    global stdp
    stdp = val
    redraw(scp, scq, ann)

def update_c(val):
    global mip2
    mip2 = val
    redraw(scp, scq, ann)

def update_d(val):
    global stdp2
    stdp2 = val
    redraw(scp, scq, ann)

def update_e(val):
    global miq
    miq = val
    redraw(scp, scq, ann)

def update_f(val):
    global stdq
    stdq = val
    redraw(scp, scq, ann)


sa.on_changed(update_a)
sb.on_changed(update_b)
sc.on_changed(update_c)
sd.on_changed(update_d)
se.on_changed(update_e)
sf.on_changed(update_f)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    sa.reset()
    sb.reset()
    sc.reset()
    sd.reset()
    se.reset()
    sf.reset()
button.on_clicked(reset)


plt.show()
