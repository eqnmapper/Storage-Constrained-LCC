# Storage Constrained Linear Computation Coding

##Run Simulations
Warning! You need at least 16 GB of RAM on your machine to run the simulations for the configured 1000 decompoisitons to average over.

To run the simulation and create the files containing the target matrices along with their decompositions run 

```
python Wiring_Script.py
```


After all the Decompositions are computed and saved under `*.npz* files, run 

```
python Error_Plot.py 
```

to create the plots. These are already uploaded under the filennames `MSE_IID.png` and `MSE_Correlated.png`.
