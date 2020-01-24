'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from radiantkit.image import ImageBinary, ImageLabeled
from radiantkit.selection import BoundingElement
from tqdm import tqdm
from typing import List, Optional, Type, Union

class ParticleSettings(object):
    _mask: Optional[ImageBinary] = None
    _roi: Optional[BoundingElement] = None

    def __init__(self):
        super(ParticleSettings, self).__init__()

    @property
    def mask(self):
        return self._mask

    @property
    def roi(self):
        return self._roi

class Particle(ParticleSettings):
    def __init__(self, B: ImageBinary, roi: BoundingElement):
        super(Particle, self).__init__()
        assert B.shape == roi.shape
        self._mask = B
        self._roi = roi

    @staticmethod
    def get_bounding_element(B: ImageBinary) -> BoundingElement:
        assert 0 == B.pixels.min() and 1 == B.pixels.max()
        axes_bounds = []
        for axis_id in range(len(B.shape)):
            axis = B.pixels.sum(tuple([axis for axis in range(len(B.shape))
                if axis != axis_id])) != 0
            axes_bounds.append((axis.argmax(), len(axis)-axis[::-1].argmax()))
        return BoundingElement(axes_bounds)

class Nucleus(Particle):
    def __init__(self, B: ImageBinary, roi: BoundingElement):
        super(Nucleus, self).__init__(B, roi)

class ParticleFinder(object):
    def __init__(self, I: Union[ImageBinary,ImageLabeled],
        particleClass: Type[Particle] = Particle):
        super(ParticleFinder, self).__init__()
        if isinstance(I, ImageBinary):
            I = ImageLabeled(I.pixels)

    @staticmethod
    def get_particles_from_labeled_image(L: ImageLabeled,
        particleClass: Type[Particle] = Particle) -> List[Type[Particle]]:
        assert L.pixels.min() != L.pixels.max(), 'monochromatic image detected.'
        particle_list = []
        for lab in tqdm(range(1, L.pixels.max())):
            B = ImageBinary(L.pixels == lab)
            roi = Particle.get_bounding_element(B)
            B = ImageBinary(roi.apply(B))
            particle_list.append(particleClass(B, roi))
        return particle_list
