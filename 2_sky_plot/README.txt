This folder contains scripts and results for the polarization sky map.

## Main script

- `final_plot.py`
Generates the polarization map using:
- our dataset (Markkanen cloud observations)
- external data (Panopoulou et al.)

## Features

- Our data and external data are plotted with different colors, so they are clearly distinguishable.
- Polarization vectors are shown on top of the dust map (`diff_ebv_gnilc_lenz.fits`).

## Outlier check

  if a visual outlier was identified in the map, do:
- find its position compared with catalog coordinates.
- change the script to check this specific position for Mark objects.
- choose which object seems to be the outlier and change the number of the object here --> if star_name == "Mark_305":

this will highlighted in red the possible outlier in the plot (polarization map) for inspection

## Output

- `polarization_map.png`
Final combined sky map with both datasets.

## Notes

- The script allows further refinement (e.g. filtering outliers or applying quality cuts).
- External data is always kept visually separate from our dataset.
