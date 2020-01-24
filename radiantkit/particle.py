'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from radiantkit.image import ImageBinary, ImageLabeled
from radiantkit.selection import BoundingElement
from typing import List, Type, Union

class ParticleSettings(object):
	def __init__(self):
		super(ParticleSettings, self).__init__()

class Particle(ParticleSettings):
	def __init__(self):
		super(Particle, self).__init__()

	@staticmethod
	def get_bounding_element(B: ImageBinary) -> BoundingElement:
		assert 0 == B.min() and 1 == B.max()
		axes_bounds = []
		for axis_id in range(len(B.shape)):
			axis = B.sum([axis for axis in range(len(B.shape))
				if axis != axis_id])
			axes_bounds.append((axis.index(1), axis[::-1].index(1)))
		return BoundingElement(axes_bounds)

class Nucleus(Particle):
	def __init__(self):
		super(Nucleus, self).__init__()

class ParticleFinder(object):
	def __init__(self, I: Union[ImageBinary,ImageLabeled],
		particleClass: Type[Particle] = Particle):
		super(ParticleFinder, self).__init__()
		if isinstance(I, ImageBinary):
			I = ImageLabeled(I.pixels)

	@staticmethod
	def get_particles_from_labeled_image(L: ImageLabeled,
		particleClass: Type[Particle] = Particle) -> List[Type[Particle]]:
		assert L.min() != L.max(), 'monochromatic image detected.'
		for lab in range(1, L.max()):
			pass