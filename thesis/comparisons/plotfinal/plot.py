import csv
import matplotlib.pyplot as plt
from collections import OrderedDict

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='csv file to plot')
    parser.add_argument('outname', type=str, help='output filename')
    parser.add_argument('-t', type=str, help='plot type', default='pdf')
    parser.add_argument('-a', type=str, help='axes', default='semilogy')
    args = parser.parse_args()
    fname = args.file

    with open(fname, 'r') as f:
        r = csv.reader(f)
        title, xlabel, ylabel = next(r)
        legend = next(r)

        xdataline = next(r)
        if len(xdataline) == 1:
            xdata = [float(x) for x in xrange(int(xdataline[0]))]
        else:
            xdata = xdataline
        data = OrderedDict()
        for line in r:
            if line[0] not in data:
                data[line[0]] = [float(line[1])]
            else:
                data[line[0]].append(float(line[1]))

    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for leg, datum in data.iteritems():
        if args.a == 'semilogy':
            plt.semilogy(xdata, datum)
        else:
            plt.plot(xdata, datum)
    plt.legend(legend)
    plt.savefig(args.outname + '.' + args.t, format=args.t)
