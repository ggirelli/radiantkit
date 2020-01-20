'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from radiantkit import path as pt

def export(path: str, exp_format: str = 'pdf') -> None:
    assert exp_format in ['pdf', 'png', 'jpg']
    path = pt.add_extension(path, '.' + exp_format)
    if exp_format == 'pdf':
        pp = PdfPages(path)
        plt.savefig(pp, format=exp_format)
        pp.close()
    else:
        plt.savefig(path, format=exp_format)
