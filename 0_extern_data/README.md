# External Polarization Data (Panopoulou et al.)

This folder contains external polarization data selected from the Panopoulou et al. (2025) catalog.

## What was done

- A standalone script (`agglomeration_polygon_selector.py`) was created based on the original Jupyter notebook.
- The script selects stars from the external catalog that fall inside a predefined sky polygon.
- Two polygon options are implemented:
- original (smaller region)
- expanded (currently used)

## Output

- `external_panopoulou_expanded_polygon.csv`
Contains the selected subset (~100 stars) within the expanded polygon.

## Notes

- The original polygon is still available in the script and can be used if needed.
- This dataset is later used in the sky plots and analysis, where it is always shown separately from our data.
