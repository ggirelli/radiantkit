'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
import sys

def main():
	parser = argparse.ArgumentParser(description = '''
	Convert a czi file into single channel tiff images.
	Output file name is either in GPSeq (default) or DOTTER notation.
	Channel names are lower-cased.

	DOTTER:  dapi_001.tif
	         <channelName>_<seriesNumber>.tif

	GPSeq:   dapi.channel001.series001.tif
	         <channelName>.channel<channelNumber>.series<seriesNumber>.tif
	''', formatter_class = argparse.RawDescriptionHelpFormatter)
	parser.add_argument('input', type = str,
	    help = '''Path to the czi file to convert.''')
	parser.add_argument('-o', '--outdir', metavar = "outdir", type = str,
	    help = """Path to output TIFF folder, created if missing. Default to a
	    folder with the input file basename.""", default = None)
	output_modes = ("DOTTER", "GPSeq")
	parser.add_argument('-m', '--mode', type = str,
	    choices = output_modes, metavar = 'mode',
	    help = """Output filename notation. Default: GPSeq.""",
	    default = "GPSeq")
	parser.add_argument('--compressed',
	    action = 'store_const', dest = 'doCompress',
	    const = True, default = False,
	    help = 'Force compressed TIFF as output.')
	version = "0.0.1"
	parser.add_argument('--version', action = 'version',
	    version = '%s %s' % (sys.argv[0], version,))
	args = parser.parse_args()
