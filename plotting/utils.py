import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

FONTSIZE = 15
ALPHA = 0.5


def format_plot(x_label, y_label, title, fontsize=None):
    if not fontsize:
        fontsize = FONTSIZE
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()


def plot_line(x, ys, contours, labels, x_label, y_label, cfg):
    for i in range(len(ys)):
        if contours:
            plt.errorbar(x, ys[i], yerr=contours[i], label=labels[i], alpha=ALPHA)
        else:
            plt.plot(x, ys[i], label=labels[i], alpha=ALPHA)
    format_plot(x_label, y_label, title='{} vs {}'.format(y_label, x_label))
    plt.savefig('{}/{}-vs-{}.png'.format(cfg.tmp_plot_folder, y_label.lower(), x_label.lower()))
    plt.close()
