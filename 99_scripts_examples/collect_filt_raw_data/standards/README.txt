
In order to create the instrument model for the given year (time interval)

cd ./standards  where the standard star measurements are collected in folders like
   BD+28.4211_1
   BD+28.4211_2
   BD+28.4211_3
   ...

run
> ls -1 > uniq_st_names.txt

then open this file uniq_st_names.txt remove suffixes like _1 and _2 then remove names of stars repeated more than once.

run
> python collect_stand.py
It will create in ./standards text files corresponding to individual stars in uniq_st_names.txt like ./standards/BD+28.4211_raw.txt
and standards_raw.dat containing data from all such files.
It will also create plots with q-u plane in each subfolder ./standards/BD+28.4211_1 that should be used to control and exclude
outliers.


Now you need to go through each subolder like ./standards/BD+28.4211_1 check x_y.png and q_u.png and remove (or just comment with #)
outliers from apertures.txt file in the corresponding folder.
After all outliers are removed (or fixed) rerun
> python collect_stand.py
and use standards_raw.dat to fit the instrument model.


This file standards_raw.dat must is used by ../../create_model/corr_QUxy_plane_unpol.py later in order to create the instrumental polarization model
