# cls2pk likelihood for MontePython 

This likelihood transforms weak lensing experiments into a Gaussian likelihood on $\alpha(k)$ as described in Doux and Karwal 2506.XXXXX . 
Please refer to the main README for the accompanying publication to cite. 

To use it, copy the contents of `likelihoods` into your MontePython directory `montepython_public/montepython/likelihoods` . Likewise, copy the contents of the directory `data` into your directory `montepython_public/data`, maintaining the folder `cls2pk` in both cases. 

Then, simply run it as shown in the `.param` file provided. You can choose the experiment you wish to run with by changing 
```
cls2pk.alpha_file  = "DES_nk24.npz"
```
to point to a different experiment, eg. `ACT_nk24.npz`. 

The noteboook `get_likelihood_components.ipynb` encodes saving the necessary likelihood components from any $\alpha(k)$ run. Please refer to that if for eg. you update the fiducial cosmology and produce all new $\alpha(k)$ to use in a MontePython likelihood. 
