import matplotlib
matplotlib.use('Qt5Agg')


import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea

mip, stdp = -4, 1
mip2, stdp2 = 4, 1
miq, stdq = 0, 1

is_double = True

xlim = -10, 10
x = np.linspace(*xlim, num=500)

def get_gauss(x, mi, var):
    return np.sqrt(1/(2*np.pi*var))*np.exp(-1/(2*var)*(x-mi)**2)

def get_kl_divergence(p, q):
    return np.sum(p * np.log2(p/q))

def get_js_divergence(p, q):
    return .5 * get_kl_divergence(p, (p+q)/2) + .5 * get_kl_divergence(q, (p+q)/2)

def get_kl_divergence_raw(p, q):
    return p * np.log2(p/q)

def get_js_divergence_raw(p, q):
    return .5 * get_kl_divergence_raw(p, (p+q)/2) + .5 * get_kl_divergence_raw(q, (p+q)/2)


def redraw(scp, scq, lpq_raw, lqp_raw, lpm_raw, lqm_raw, ljs, ann):
    if is_double:
        yp = .5*get_gauss(x, mip, stdp**2) + .5*get_gauss(x, mip2, stdp2**2)
    else:
        yp = get_gauss(x, mip, stdp**2)
    scp.set_offsets(np.c_[x,yp])
    yq = get_gauss(x, miq, stdq**2)
    scq.set_offsets(np.c_[x,yq])
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
    kl_js = get_js_divergence(yq, yp)
    if kl_js < 1:
        kl_jsstr = f'{kl_qp:.1f}'
    else:
        kl_jsstr = f'{kl_qp:.0f}'
    ann.set_text(f'kl_pq: {kl_pqstr}, kl_qp: {kl_qpstr}, kl_js: {kl_jsstr}')

    kl_pq_raw = get_kl_divergence_raw(yp, yq)
    kl_qp_raw = get_kl_divergence_raw(yq, yp)

    lpq_raw.set_data(x,kl_pq_raw)
    lqp_raw.set_data(x,kl_qp_raw)

    m = (yp + yq) / 2

    kl_pm_raw = get_kl_divergence_raw(yp, m)
    kl_qm_raw = get_kl_divergence_raw(yq, m)

    lpm_raw.set_data(x,kl_pm_raw)
    lqm_raw.set_data(x,kl_qm_raw)

    ljs.set_data(x, get_js_divergence_raw(yp, yq))

    fig.canvas.draw_idle()




# init plot
fig, ax = plt.subplots(2, 2)
plt.subplots_adjust(left=0.2, bottom=0.25)

ax0 = ax[0,0]
yp = .5*get_gauss(x, mip, stdp**2) + .5*get_gauss(x, mip2, stdp2**2)
scp = ax0.scatter(x, yp, s=1)
yq = get_gauss(x, miq, stdq**2)
scq = ax0.scatter(x, yq, s=1)
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
kl_js = get_js_divergence(yq, yp)
if kl_js < 1:
    kl_jsstr = f'{kl_qp:.1f}'
else:
    kl_jsstr = f'{kl_qp:.0f}'
ann = ax0.annotate(f'kl_pq: {kl_pqstr}, kl_qp: {kl_qpstr}, kl_js: {kl_jsstr}', xy=(0, 1), xycoords="axes fraction",
        va="top", ha="left",
        bbox=dict(boxstyle="round", fc="w"), fontsize=20)
ax0.set_ylim(0, 1)
ax0.legend(['p(x)', 'q(x)'])
ax0.set_title('raw')

ax1 = ax[0,1]
kl_pq_raw = get_kl_divergence_raw(yp, yq)
kl_qp_raw = get_kl_divergence_raw(yq, yp)

lpq_raw = ax1.plot(x, kl_pq_raw)[0]
lqp_raw = ax1.plot(x, kl_qp_raw)[0]
ax1.legend(['kl_pq', 'kl_qp'])
ax1.set_title('kullback-leibler 1')

ax2 = ax[1,0]
m = (yp + yq) / 2
kl_pm_raw = get_kl_divergence_raw(yp, m)
kl_qm_raw = get_kl_divergence_raw(yq, m)

lpm_raw = ax2.plot(x, kl_pm_raw)[0]
lqm_raw = ax2.plot(x, kl_qm_raw)[0]
ax2.legend(['kl_pm', 'kl_qm'])
ax2.set_title('kullback-leibler 2')

ax3 = ax[1,1]
ljs = ax3.plot(x, get_js_divergence_raw(yp, yq))[0]
ax3.set_title('jensen-shannon')

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


rax = plt.axes([0.05, 0.7, 0.1, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('single', 'double'),active=1)

def radio_fun(label):
    global is_double
    is_double = True if label == 'double' else False
    redraw(scp, scq, lpq_raw, lqp_raw, lpm_raw, lqm_raw, ljs, ann)

radio.on_clicked(radio_fun)


def update_a(val):
    global mip
    mip = val
    redraw(scp, scq, lpq_raw, lqp_raw, lpm_raw, lqm_raw, ljs, ann)

def update_b(val):
    global stdp
    stdp = val
    redraw(scp, scq, lpq_raw, lqp_raw, lpm_raw, lqm_raw, ljs, ann)

def update_c(val):
    global mip2
    mip2 = val
    redraw(scp, scq, lpq_raw, lqp_raw, lpm_raw, lqm_raw, ljs, ann)

def update_d(val):
    global stdp2
    stdp2 = val
    redraw(scp, scq, lpq_raw, lqp_raw, lpm_raw, lqm_raw, ljs, ann)

def update_e(val):
    global miq
    miq = val
    redraw(scp, scq, lpq_raw, lqp_raw, lpm_raw, lqm_raw, ljs, ann)

def update_f(val):
    global stdq
    stdq = val
    redraw(scp, scq, lpq_raw, lqp_raw, lpm_raw, lqm_raw, ljs, ann)


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
