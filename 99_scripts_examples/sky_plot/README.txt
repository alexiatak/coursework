

The plot is poduced in the last cell of Sample_skyplot.ipynb

Some libraries were incompatible with 'mark' conda environment so i created another one 'mark_skyplot'
specifically to run this notebook. Most likely it can be done different way (maybe just downgrading python version).
For now you can install this environment
conda env create -f mark_skyplot_env.yml

if it dosn't work even in this environment, try the following
conda create -n mark_skyplot python=3.11
conda install  -c conda-forge  healpy matplotlib astropy pygplates jupyter


Files that used are:

gaia_stars.pickle                    - all stars from Gaia in this region down to 17mag (was needed during samle selection but not anymore)
mark_2025.png                        - the sky plot with positions of our stars
Markkanen_schedule.csv               - stars remaining to be observed
Markkanen_schedule_original.csv      - all sample stars
observing_list_entire.txt            - same stars as in Markkanen_schedule_original.csv but without names but with GaiaIDs
observing_list_small.txt             - initial sample selected from Gaia
Sample_skyplot.pdf                   - printout of Sample_skyplot.ipynb aved as pdf
