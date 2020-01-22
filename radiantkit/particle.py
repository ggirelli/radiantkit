'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

class ParticleSettings(object):
	def __init__(self):
		super(ParticleSettings, self).__init__()

class Particle(ParticleSettings):
	def __init__(self, *args, **kwargs):
		super(Particle, self).__init__(*args, **kwargs)

class Nucleus(Particle):
	def __init__(self, *args, **kwargs):
		super(Nucleus, self).__init__(*args, **kwargs)
