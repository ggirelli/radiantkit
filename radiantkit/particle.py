'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

class ParticleSettings(object):
	"""docstring for ParticleSettings"""
	def __init__(self):
		super(ParticleSettings, self).__init__()

class Particle(ParticleSettings):
	"""docstring for Particle"""
	def __init__(self, *args, **kwargs):
		super(Particle, self).__init__(*args, **kwargs)

class Nucleus(Particle):
	"""docstring for Nucleus"""
	def __init__(self, *args, **kwargs):
		super(Nucleus, self).__init__(*args, **kwargs)
