from setuptools import setup
from version import get_git_version

setup(name='dsa110-nsfrb',
      #version=get_git_version(),
      version="0.1.0",
      url='http://github.com/dsa110/dsa110-nsfrb',
      python_requires='>3.8',
#      requirements=['seaborn', 'astropy', 'hdbscan', 'progress'],
#      packages=['nsfrb'],
      zip_safe=False)
