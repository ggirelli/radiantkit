'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from radiantkit.image import Image
from typing import Dict

class SeriesSettings(object):
	"""docstring for SeriesSettings"""
	def __init__(self, arg):
		super(SeriesSettings, self).__init__()
		self.arg = arg

class Series(SeriesSettings):
	"""docstring for Series"""

	__channels: Dict[str, Image] = {}

	def __init__(self, *args, **kwargs):
		super(Series, self).__init__(*args, **kwargs)
