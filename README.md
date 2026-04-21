# P-FLARE --- CPU version
Penalised FLux-driven Algorithm for REduced models https://doi.org/10.1017/S0022377825100895

Some animations of numerical experiments with the Hasegawa-Wakatani model are available at https://zenodo.org/records/15423476

## Dowload the CPU version of the code
```code
git clone -b CPU https://github.com/piergui/P-FLARE.git
```

## Run the code
Default example with a particle source, with C=1.0 and starting from an initial profile with kap=1.2 so that we are in the zonal flow dominated regime 
```code
python run_P-FLARE.py
```

## Post-process the output file
The jupyter-notebook `POST-PROCESS_P-FLARE.ipynb` contains a basic post-processing of the ouput file
