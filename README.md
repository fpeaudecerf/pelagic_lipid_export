# Model for bacterial degradation of lipids in sinking marine snow

This repository contains the code associated with the theoretical model of bacterial degradation of lipids in marine snow particles sinking through the water column, and the associated computations of transfer efficiencies, as presented in the manuscript:

“Microbial metabolic specificity controls pelagic lipid export efficiency” Lars Behrendt, Uria Alcolombri, Jonathan E. Hunter, Steven Smriga, Tracy Mincer, Daniel P. Lowenstein, Yutaka Yawata, François J. Peaudecerf, Vicente I. Fernandez, Helen F. Fredricks, Henrik Almblad, Joe J. Harrison, Roman Stocker, Benjamin A. S. Van Mooy, *BioRxiv* (2024)\
DOI of manuscript: 10.1101/2023.12.08.570822 \
Direct link to manuscript: https://www.biorxiv.org/content/10.1101/2023.12.08.570822v1

The files are organised as follows:
- **oil_model.py** contains the chosen numerical parameters from the manuscript (as given in the Supplementary “Full description of the lipid vertical flux”), all the functions needed to compute the dynamics of degradation and sinking of marine snow particles composed of ballast and oil as presented in the model, and the functions needed to compute transfer efficiencies of lipid fluxes at different depths. These parameters and functions are used at the end of the script to generate a visualisation of results corresponding to Supplementary figure 8 of the manuscript.
- **monte_carlo_rates.csv** contains results of a Monte-Carlo resampling simulation run from experimental measurements in the lab in order to obtain a range of typical degradation rate constants k_A  (see supplementary manuscript “Full description of the lipid vertical flux”). These results are presented in the Supplementary figure 8 and thus are needed by oil_model.py to generate this figure.
- **utils.py** is a small utilities file.
- **sinking_dynamics_example_oil_model.eps** is an example output of oil_model.py representing on a figure the depth reached by particles of different initial sizes with time.
- **FigSF_full.eps** is an exemple output of oil_model.py corresponding to Supplementary figure 8 of the manuscripted referenced above.
- **environment.yml** sets the environment for Binder, so that one can directly execute the script contained in oil_model.py in Binder and have it work there, without the need for a local Python installation.

To run the model in Binder without the need of a local Python installation, click on the blue badge above or following the URL below:\
[URL to add]

There, you can select “New” -> “Terminal” and execute the scripts from there.
