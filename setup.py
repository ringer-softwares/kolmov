# Imports
from setuptools import setup, find_packages

# Loading requirements
deps = []
with open('requirements.txt', 'r') as f:
  for line in f.readlines():
    deps.append(line.rstrip("\n"))

# Loading README file
with open("README.md", "r") as fh:
  long_description = fh.read()

setup(
  name = 'kolmov',
  version = '1.0.0',
  license='GPL-3.0',
  description = 'A Framework to perfomance the cross validation for Ringer tunings',
  long_description = long_description,
  long_description_content_type="text/markdown",
  packages=find_packages(),
  author = 'Micael Veríssimo de Araújo, Gabriel Gazola Milan, João Victor da Fonseca Pinto',
  author_email = 'micael.verissimo@lps.ufrj.br, gabriel.milan@lps.ufrj.br, jodafons@lps.ufrj.br',
  url = 'https://github.com/micaelverissimo/kolmov',
  keywords = ['framework', 'validation', 'machine-learning', 'ai', 'plotting', 'data-visualization'],
  install_requires=deps,
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)