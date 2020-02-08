'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from setuptools import setup, find_packages
from distutils.util import convert_path
from codecs import open
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

const_data = {}
ver_path = convert_path('radiantkit/const.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), const_data)

setup(
    name='radiantkit',
    version=const_data['__version__'],
    description='Radial Image Analysis Toolkit',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ggirelli/gpseq-img-py',
    author='Gabriele Girelli',
    author_email='gabriele.girelli@scilifelab.se',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='microscopy image analysis bioimaging biology cell DNA',
    packages=find_packages(),
    install_requires=[
        'czifile==2019.7.2',
        'ggc==0.0.3',
        'jinja2==2.10.3',
        'joblib==0.14.1',
        'matplotlib==3.1.2',
        'nd2reader==3.2.3',
        'numpy==1.18.1',
        'pandas==0.25.3',
        'plotly==4.5.0',
        'scikit-image==0.16.2',
        'scipy==1.4.1',
        'seaborn==0.9.0',
        'tifffile==2019.7.26.2',
        'tqdm==4.41.1'
    ],
    scripts=[],
    entry_points={'console_scripts': [
        'radiant = radiantkit.scripts.radiant:main']},
    test_suite='nose.collector',
    tests_require=['nose']
)
