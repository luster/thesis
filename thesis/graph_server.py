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
        loss = []
        mse = []
        floss = []
        fmse = []
        for line in r:
            if line[3] == 'pretrain':
                loss.append(float(line[1]))
                mse.append(float(line[2]))
            else:
                floss.append(float(line[1]))
                fmse.append(float(line[2]))
        fig = plt.figure(figsize=(32,16), dpi=80)
        # plt.title('{}%'.format(progress))
        plt.subplot(411)
        plt.plot(loss)
        plt.title('loss')
        plt.subplot(412)
        plt.plot(mse)
        plt.title('mse')
        plt.subplot(413)
        plt.plot(floss)
        plt.title('f_loss')
        plt.subplot(414)
        plt.plot(fmse)
        plt.title('f_mse')
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
    regex = '^out_(noisy|Scc|Sdc|Scd|Sdd)_-{0,1}\d{1,2}\.\d.wav$'
    fname = 'out_{}_{}.wav'.format(reconstruction, snr)
    print fname
    if not re.match(regex, fname):
        return make_response('Forbidden')
    fpath = os.path.join(latest, fname)
    return send_file(fpath, mimetype='audio/wav', as_attachment=False)

@app.route('/play/<fname>')
def snd(fname):
    if fname not in ['xhat', 'noisy', 'clean', 'Scc', 'fine_xhat', 'wtf']:
        return make_response('Forbidden')
    d = os.path.join(get_latest_sim_dir(), 'wav')
    print d
    fpath = os.path.join(d, '{}.wav'.format(fname))
    return send_file(fpath, mimetype='audio/wav', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
