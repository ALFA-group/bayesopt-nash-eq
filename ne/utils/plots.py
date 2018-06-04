import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import matplotlib
import json
import pandas as pd
import itertools

matplotlib.rcParams['axes.formatter.useoffset'] = False
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
FONTSIZE = 12.5
COLORS = itertools.cycle(["#a80000", "#00a8a8", "#5400a8", "#54a800",
                          '#dc00dc', '#dc6e00', '#00dc00', '#006edc'])
MARKERS = itertools.cycle(['.', '+', 'o', '*', 'v', '>', '<', 'd'])
plt.rcParams["mathtext.fontset"] = "cm"

markers_dict = {
    'GPG-sur': '>',
    'GPG-psim': '<',
    'BN-approx': '*',
    'BN-exact': 'd',
    'BR': '.'
}

colors_dict = {
    'GPG-sur': '#a80000',
    'GPG-psim': '#5400a8',
    'BN-approx': '#54a800',
    'BN-exact': '#dc00dc',
    'BR': '#dc6e00'
}

names_dict = {
    'GPG-sur': r'\texttt{GPG-sur}',
    'GPG-psim': r'\texttt{GPG-psim}',
    'BN-approx': r'\texttt{BN-approx}',
    'BN-exact': r'\texttt{BN-exact}',
    'BR': r'\texttt{BR}'
}


def plot_contour(_f):
    x, y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    fxy = map(np.vectorize(_f), x, y)
    plt.pcolormesh(x, y, fxy, cmap=plt.cm.Blues)
    plt.show()


def plot_marginalized_responses(pf, mf_over_p, mf_over_others, rf, p=1, title='mar-response', actual_ne=None):
    """

    :param title:
    :param actual_ne:
    :param pf: payoff function
    :param mf_over_p: marginalized payoff over p
    :param mf_over_others: marginalized payoff over {-p}
    :param rf: regret function for p
    :param p: player id
    :return:
    """
    p = p + 1
    num_pts = 50
    x, y = np.meshgrid(np.linspace(0, 1, num_pts), np.linspace(0, 1, num_pts))
    f_vals = map(np.vectorize(lambda _x, _y: pf([_x, _y])), x, y)
    r_vals = map(np.vectorize(lambda _x, _y: rf([_x, _y])), x, y)

    def exp_fctx(_x):
        mu, _ = mf_over_p(_x)
        return mu

    def exp_fcty(_y):
        mu, _ = mf_over_others(_y)
        return mu

    def std_fctx(_x):
        _, std = mf_over_p(_x)
        return std

    def std_fcty(_y):
        _, std = mf_over_others(_y)
        return std

    m_x = np.array(map(exp_fctx, x[0, :]))
    m_y = np.array(map(exp_fcty, y[:, 0]))
    std_x = np.array(map(std_fctx, x[0, :]))
    std_y = np.array(map(std_fcty, y[:, 0]))

    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    # first response
    im = ax.pcolormesh(x, y, f_vals, cmap=plt.cm.Blues)
    ax.set_title('$\mu_%d(x_1,x_2\mid \mathcal{D}^{t})$' % p, fontsize=FONTSIZE, weight='bold')

    ax.set_aspect(1.)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)

    def format_coord(_x, _y, _ax=None, z=None):
        x0, x1 = _ax.get_xlim()
        y0, y1 = _ax.get_ylim()
        col = int(np.floor((_x - x0) / float(x1 - x0) * num_pts))
        row = int(np.floor((_y - y0) / float(y1 - y0) * num_pts))
        if 0 <= col < num_pts and 0 <= row < num_pts:
            _z = z[row][col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (_x, _y, _z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = lambda _x, _y: format_coord(_x, _y, z=f_vals, _ax=ax)

    # the rest
    divider = make_axes_locatable(ax)
    ax_mx = divider.append_axes("bottom", 1, pad=0.5, sharex=ax)
    ax_my = divider.append_axes("left", 1, pad=0.55, sharey=ax)
    ax_mr = divider.append_axes("right", 2.5, pad=0.45, sharey=ax)
    ax_cb = divider.append_axes("top", 0.25, pad=0.65)
    # color bar
    plt.colorbar(im, cax=ax_cb, orientation="horizontal")
    ax_cb.xaxis.set_major_locator(MaxNLocator(2))
    ax_cb.set_xticklabels(['Low', 'Medium', 'High'])
    #
    ax_mx.xaxis.set_tick_params(labelbottom=False)
    ax_my.yaxis.set_tick_params(labelleft=False)

    ax_mx.yaxis.set_ticks_position("right")
    ax_my.xaxis.set_ticks_position("top")
    ax_my.xaxis.set_tick_params(rotation=90)

    ax_mx.plot(x[0, :], m_x, 'k')
    ax_mx.set_ylabel('$E_{x_2}\mu_%d(x_1,x_2\mid \mathcal{D}^{t})$' % p, fontsize=FONTSIZE, weight='bold')
    ax_mx.fill(np.concatenate([x[0, :], x[0, ::-1]]), np.concatenate([m_x - 0.5 * std_x, (m_x + 0.5 * std_x)[::-1]]),
               alpha=.5, fc='b', ec='None', label='95% CI')

    ax_my.fill(np.concatenate([m_y - 0.5 * std_y, (m_y + 0.5 * std_y)[::-1]]), np.concatenate([y[:, 0], y[::-1, 0]]),
               alpha=.5, fc='b', ec='None', label='95% CI')
    ax_my.plot(m_y, y[:, 0], 'k')
    ax_my.set_xlabel('$E_{x_1}\mu_%d(x_1,x_2\mid \mathcal{D}^{t})$' % p, fontsize=FONTSIZE, weight='bold')
    ax_my.set_xlim(ax_my.get_xlim()[::-1])

    ax_mr.pcolormesh(x, y, r_vals, cmap=plt.cm.Blues)
    ax_mr.format_coord = lambda _x, _y: format_coord(_x, _y, z=r_vals, _ax=ax_mr)
    ax_mr.set_title('$\hat{r}_%d$' % p, fontsize=FONTSIZE, weight='bold')
    ax_mr.set_aspect(1.)
    ax_mr.set_xlabel('$x_1$', fontsize=12)
    ax_mr.set_ylabel('$x_2$', fontsize=12)

    # plot ne stuff
    if actual_ne is not None:
        ax.scatter(actual_ne[0], actual_ne[1], c="r", marker="^", label='NE', alpha=0.5)
        ax_mr.scatter(actual_ne[0], actual_ne[1], c="r", marker="^", alpha=0.5)
        ax.axhline(actual_ne[1], color='r', linestyle='--', alpha=0.5)
        ax.axvline(actual_ne[0], color='r', linestyle='--', alpha=0.5)
        ax_my.axhline(actual_ne[1], color='r', linestyle='--', alpha=0.5)
        ax_mx.axvline(actual_ne[0], color='r', linestyle='--', alpha=0.5)

    ax.legend()
    ax_my.legend()
    ax_mx.legend()

    if title is None:
        plt.show()
    else:
        plt.savefig(title + '.pdf')


def plot_objective_space(_f, f_vals, actual_ne=None, title='obj-space'):
    plt.clf()
    import itertools
    x, y = np.meshgrid(np.linspace(0, 1, 250), np.linspace(0, 1, 250))
    f1 = list(itertools.chain(*map(np.vectorize(lambda _x, _y: _f([_x, _y], is_minimize=True)[0]), x, y)))
    f2 = list(itertools.chain(*map(np.vectorize(lambda _x, _y: _f([_x, _y], is_minimize=True)[1]), x, y)))
    plt.scatter(f1, f2, c="k", label="Objective Space")
    plt.scatter(*(zip(*f_vals)), c="g", label="Observations")
    if actual_ne is not None:
        plt.scatter(actual_ne[0], actual_ne[1], c="r", marker="^", label='NE')
    plt.xlabel('$f_1$', fontsize=FONTSIZE)
    plt.ylabel('$f_2$', fontsize=FONTSIZE)
    plt.legend()
    plt.tight_layout()
    if title is None:
        plt.show()
    else:
        plt.savefig(title + '.pdf')


def plot_decision_space(pts, actual_ne=None, title='dec-space'):
    x, y = zip(*pts)
    g = sns.jointplot(x=np.array(x), y=np.array(y), kind="kde", color="k")
    g.plot_joint(plt.scatter, c="g", s=10, linewidth=1, marker="*", label="Observations")

    g.ax_joint.collections[0].set_alpha(0.1)
    if actual_ne is not None:
        plt.scatter(actual_ne[0], actual_ne[1], c="r", marker="^", label='NE', alpha=0.5)
    plt.xlim([-0.5, 1.5])
    plt.ylim([-0.5, 1.5])
    g.set_axis_labels(xlabel="$x_1$", ylabel="$x_2$")
    plt.legend()
    plt.tight_layout()
    if title is None:
        plt.show()
    else:
        plt.savefig(title + '.pdf')


def plot_regret_trace(data_in, is_file=True):
    """

    :param json_fname:
    :return:
    """
    if is_file:
        json_fname = data_in
        # load file
        if isinstance(json_fname, list):
            json_fnames = json_fname
            data = []
            for json_fname in json_fnames:
                with open(json_fname, 'r') as _f:
                    data.append(json.load(_f))
        else:
            with open(json_fname, 'r') as _f:
                data = json.load(_f)
    else:
        data = data_in

    regret_gb_obj = pd.DataFrame(data).groupby('alg').regret_trace

    # handy vars
    ax = plt.subplot()

    max_xlim = np.inf
    for alg, traces in regret_gb_obj:
        traces = traces.values.tolist()
        len_shrtst_trace = min([len(trace) for trace in traces])
        max_xlim = min(max_xlim, len_shrtst_trace)
        rs = np.array([trace[:len_shrtst_trace] for trace in traces])
        rs[rs <= 0] = 1e-20  # np.finfo(np.float32).eps
        mean_rs = np.mean(rs, 0)
        std_rs = np.std(rs, 0)
        color = colors_dict[alg]  # COLORS.next()
        marker = markers_dict[alg]  # MARKERS.next()
        alg_name = names_dict[alg]
        # plt.errorbar(range(len_shrtst_trace), np.log(mean_rs), yerr=np.log(std_rs), label=alg, fmt='--o')
        plt.plot(range(1, len_shrtst_trace + 1), np.log(mean_rs), '--' + marker, label=alg_name, color=color)
        plt.fill_between(range(1, len_shrtst_trace + 1), np.log(mean_rs) - 0.434 * std_rs / mean_rs,
                         np.log(mean_rs) + 0.434 * std_rs / mean_rs,
                         alpha=0.2, facecolor=color)

    # setup
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel('$\\boldsymbol{\log(\epsilon(\mathbf{x}))}$', fontsize=1.5 * FONTSIZE)
    plt.xlabel(r'$\boldsymbol{\texttt{\#FEs}}$', fontsize=1.4 * FONTSIZE
               )
    plt.xlim([1, max_xlim])
    plt.legend(fontsize=FONTSIZE, loc=3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # some regex
    from glob import glob
    import re

    is_batch = False

    if is_batch:
        regex_expr = r"saddle_res_3_[^_]*_2.*"
        regex_expr = r"saddle_res_[^35_]*_2.*"
        files = [f for f in glob('../experiments/res/saddle_res_*') if re.search(regex_expr, f)]
        print(files)
        plot_regret_trace(files)
    else:
        plot_regret_trace('../experiments/res/mop_res_25runs.json')
