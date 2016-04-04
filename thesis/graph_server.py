from flask import Flask, send_file, make_response
import matplotlib.pyplot as plt
import os
import glob
import csv
import re
from StringIO import StringIO
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)


def get_latest_sim_dir():
    return max(glob.glob(os.path.join('sim', '*/')), key=os.path.getmtime)


@app.route('/')
def generate_graph():
    latest = get_latest_sim_dir()
    fname = os.path.join(latest, 'graphs.txt')
    with open(fname, 'r') as f:
        r = csv.reader(f)
        loss_mag = []
        loss_phase = []
        mse_dc = []
        mse_dd = []
        for line in r:
            loss_mag.append(line[1])
            loss_phase.append(line[2])
            mse_dc.append(line[3])
            mse_dd.append(line[4])
        progress = int(line[0])/768. * 100
        fig = plt.figure(figsize=(16,16), dpi=80)
        # plt.title('{}%'.format(progress))
        plt.subplot(411)
        plt.plot(loss_mag)
        plt.title('loss_mag progress = %d%%' % (progress))
        plt.subplot(412)
        plt.plot(loss_phase)
        plt.title('loss_phase')
        plt.subplot(413)
        plt.plot(mse_dc)
        plt.title('mse_dc')
        plt.subplot(414)
        plt.plot(mse_dd)
        plt.title('mse_dd')
        canvas = FigureCanvas(fig)
        tmp = StringIO()
        canvas.print_png(tmp)
        plt.savefig(tmp, format='png')
    response = make_response(tmp.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response


@app.route('/sound/<reconstruction>/<snr>')
def play_sound(reconstruction, snr):
    latest = os.path.join(get_latest_sim_dir(), 'wav')
    regex = 'out_(noisy|Scc|Sdc|Scd|Sdd)_-{0,1}\d\.\d.wav'
    fname = 'out_{}_{}.wav'.format(reconstruction, snr)
    if not re.match(regex, fname):
        return
    fpath = os.path.join(latest, fname)
    return send_file(fpath, mimetype='audio/wav', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
