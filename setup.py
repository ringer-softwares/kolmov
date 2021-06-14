# Imports
from setuptools import setup, find_packages

# Loading README file
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='kolmov',
    version='2.0.4',
    license='GPL-3.0',
    description='A Framework for performing cross validation for Ringer tunings',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    author='Micael Veríssimo de Araújo, Gabriel Gazola Milan, João Victor da Fonseca Pinto',
    author_email='micael.verissimo@lps.ufrj.br, gabriel.milan@lps.ufrj.br, jodafons@lps.ufrj.br',
    url='https://github.com/micaelverissimo/kolmov',
    keywords=['framework', 'validation', 'machine-learning',
              'ai', 'plotting', 'data-visualization'],
    install_requires=[
        "numpy>=1.16.6,<2.0a0",
        "pandas==1.2.4",
        "matplotlib==3.4.2",
        "seaborn==0.11.1",
        "Gaugi>=1.0.0",
        "tensorflow==2.5.0",
        "scikit-learn==0.24.2",
        "onnxruntime==1.8.0",
        "keras2onnx==1.7.0"
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
