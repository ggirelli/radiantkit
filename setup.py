"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
import os

here = os.path.abspath(os.path.dirname(__file__))
bindir = os.path.join(here, 'bin/')

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
	long_description = f.read()

setup(name='radiantkit',
	version='0.0.1',
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
		'scikit-image==0.16.2',
		'scipy==1.4.1',
		'seaborn==0.9.0',
		'tifffile==2019.7.26.2',
		'tqdm==4.41.1'
	],
	scripts=[],
	test_suite='nose.collector',
	tests_require=['nose'],
	entry_points={'console_scripts':[
		'czi_to_tiff = radiantkit.scripts.czi_to_tiff:main',
		'nd2_to_tiff = radiantkit.scripts.nd2_to_tiff:main',
		'tiff_findoof = radiantkit.scripts.tiff_findoof:main',
		'tiff_segment = radiantkit.scripts.tiff_segment:main',
		'tiff_split = radiantkit.scripts.tiff_split:main',
		'tiff_desplit = radiantkit.scripts.tiff_desplit:main',
		'tiffcu = radiantkit.scripts.tiffcu:main'
	]}
)
