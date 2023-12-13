from setuptools import setup
from version import get_git_version

setup(name='dsa110-nsfrb',
      version=get_git_version(),
      url='http://github.com/dsa110/dsa110-nsfrb',
#      requirements=['seaborn', 'astropy', 'hdbscan', 'progress'],
#      packages=['nsfrb'],
      zip_safe=False)
