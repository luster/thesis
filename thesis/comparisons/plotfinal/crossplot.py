import re
import glob
import csv
import matplotlib.pyplot as plt
from collections import OrderedDict

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=10)
params = {'legend.fontsize': 6,
          }
plt.rcParams.update(params)

"""
we want a plot at
    specified snr
    all unique prefixes
    mse OR loss plot
    legend??
"""


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, help='mse or loss')
    parser.add_argument('snr', type=int, help='snr to use in plot')
    args = parser.parse_args()
    if args.type not in ('mse', 'loss',):
        raise Exception('type should be "mse" or "loss"')
    fnames = glob.glob("*{}.csv".format(args.type))
    print fnames

    title = 'Comparison of {} of various networks at {} dB'.format(args.type, args.snr)
    xlabel = 'Iterations'
    ylabel = 'MSE' if args.type == 'mse' else args.type
    regex = r'^(.*)-{}.csv$'.format(args.type)
    legends = []
    data = OrderedDict()
    snrfmt = '{} dB'.format(args.snr)
    for fname in fnames:
        data[fname] = []
        legend = re.match(regex, fname).groups()[0]
        legends.append(legend)

        with open(fname, 'r') as f:
            r = csv.reader(f)
            next(r)
            next(r)
            xdataline = next(r)
            if len(xdataline) == 1:
                xdata = [float(x) for x in xrange(int(xdataline[0]))]
            else:
                xdata = xdataline

            for row in r:
                snr = row[0]
                if snr != snrfmt:
                    continue

                data[fname].append(float(row[1]))

    plt.figure(figsize=(4.5,3), dpi=300)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)#, rotation=0)
    plt.title(title)
    for key, datum in data.iteritems():
        plt.semilogy(xdata,datum)
    plt.legend(legends)
    outfname = 'pdf/comparison-{}-{}.pdf'.format(args.type,args.snr)
    plt.tight_layout()
    plt.savefig(outfname, format='pdf', bbox_inches='tight')
