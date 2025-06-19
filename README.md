# Going beyond $S_8$: fast inference of the matter power spectrum from weak-lensing surveys

This repository contains the [data](data), code ([cls2pk.py](cls2pk.py)) and notebook ([cls2pk.ipynb](cls2pk.ipynb)) associated with the paper *Going beyond* $S_{8}$ *: fast inference of the matter power spectrum from weak-lensing surveys* (2025) by Cyrille Doux and Tanvi Karwal. The paper extracts the matter power spectrum $P(k,z)$ from tomographic weak-lensing surveys (DES, KiDS, HSC and ACT) to test the ΛCDM cosmological model and identify potential deviations from Planck CMB data expectations.

The notebook demonstrates the steps involved to reproduce the main results from this analysis, including:
* Loading and pre-processing weak-lensing data (results saved [here](res)).
* Extracting the matter power spectrum $P(k, z)$ with fast inference using Hamiltonian Monte Carlo (chains saved [here](chains)).
* Comparing the results with Planck ΛCDM expectations.
* Investigating extensions of the ΛCDM model predictions for the matter power spectrum.

In addition, the [likelihood](likelihood) folder contains derived likelihoods ready to use with the [MontePython](https://baudren.github.io/montepython.html) inference framework as well as code to generate these likelihoods at different fiducial cosmologies.

