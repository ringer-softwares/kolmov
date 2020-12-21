
[![Build Status](https://travis-ci.org/micaelverissimo/kolmov.svg?branch=master)](https://travis-ci.org/micaelverissimo/kolmov)

# KoLMoV

## Introduction:

In 2017 the ATLAS experiment implemented an ensemble of neural networks (NeuralRinger algorithm) dedicated to improving the performance of filtering events containing electrons in the high-input rate online environment of the Large Hadron Collider at CERN, Geneva. The ensemble employs a concept of calorimetry rings. The training procedure and final structure of the ensemble are used to minimize fluctuations from detector response, according to the particle energy and position of incidence. This reposiroty is dedicated to hold all analysis scripts for each subgroup in the ATLAS e/g trigger group.

>**NOTE** This repository make part of the ringer analysis kit.

## What does mean?

KoLMoV (**K**it **o**f **L**earning **M**odels **V**alidation) is a repository that contains somes helpers to calculate the cross validation or pileup linear fit for ringer tunings.

## How to install?

Install from pip:
```bash
pip install kolmov
```
or from the source:
```bash
git clone https://github.com/micaelverissimo/kolmov.git 
cd kolmov
source scripts/setup.sh
```
>**WARNING**: You must have ROOT installed at your system.






