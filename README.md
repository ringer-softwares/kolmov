# KoLMoV

[![PyPI Version](https://img.shields.io/pypi/v/kolmov)](https://pypi.org/project/kolmov/)
[![Python Versions](https://img.shields.io/pypi/pyversions/kolmov)](https://github.com/jodafons/kolmov)

We should include some description here.

## What does mean?

KoLMoV (**K**it **o**f **L**earning **M**odels **V**alidation) is a repository that contains somes helpers to calculate the cross validation or pileup linear fit for ringer tuning derived from [saphyra](https://github.com/ringer-atlas/saphyra) package.

**NOTE** This repository is part of the ringer analysis kit.

## Installation:

Install stable version from pip:
```bash
pip install kolmov
```
or install latest version from git:
```bash
pip install git+https://github.com/ringer-atlas/kolmov.git@master
```
or install from source:
```bash
git clone https://github.com/ringer-atlas/kolmov.git 
cd kolmov
source scripts/setup.sh
```

## Notes about ringer project:

In 2017 the ATLAS experiment implemented an ensemble of neural networks (NeuralRinger algorithm) dedicated to improving the performance of filtering events containing electrons in the high-input rate online environment of the Large Hadron Collider at CERN, Geneva. The ensemble employs a concept of calorimetry rings. The training procedure and final structure of the ensemble are used to minimize fluctuations from detector response, according to the particle energy and position of incidence.





