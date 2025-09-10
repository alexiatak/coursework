
Scripts needed for data cleaning and preparison

1. Collect data as shown in collect_filt_raw_data
2. Find instrumental polarization corrections as shown in create_model (you can skip it - already done)
3. Correct Markkanen data using the model found in the previous step in correct_instrumental
4. Markkanen cloud sky gas-to-dust map is in sky_plot


Multiband data
1. In multiband/Correct_R.ipynb the instrumental polarization correction in the R band is demonstrated
   with fake data. I should be repeated the same way for all 4 bands.
