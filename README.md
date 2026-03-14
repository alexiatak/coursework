Workflow
-------------------------------------------------------------------------------------------------
Environment --> Activate the working conda environment before running the scripts:
conda activate mark

Step 1 — Add raw star data
Add star folders to: coursework/0_raw_data/R/Markkanen/

Each star should have its own folder, for example:
Mark_24
Mark_25
Mark_26

If a star has multiple independent observation series, divide them into separate folders:
Mark_26
Mark_26B
Mark_26C

Inside each star folder, place all FITS files inside a subfolder named by Julian date. Example:
Mark_26/
└── 2460312/
    ├── file1.fits
    ├── file2.fits

⸻

Step 2 — Set parameters in proc_RBPL.py

Open the FITS files and determine whether the target star is bright or faint.
Set the display parameter in proc_RBPL.py:

For bright objects: LEVELS = 80
For faint objects: LEVELS = 99

You may also adjust the following parameters if needed:

aperts = [...] to  change aperture sizes.
MANUAL_FOR_ALL_SHOTS = True  to manually click the object center for every FITS file.
USE_SEP = True  to enable automatic source detection.

⸻

Step 3 — Run proc_RBPL.py

Run the processing script:
conda activate mark
python proc_RBPL.py

A figure containing four objects will appear.Click the centers in the following order:0 → 1 → 2 → 3
Then click the scroll wheel / middle mouse button to submit.

⸻

Step 4 — Select the best aperture

For every FITS file, the script creates a folder containing five PNG images with different aperture sizes.
Inspect these images and choose which aperture best matches the star. Write the selected aperture in: apertures.txt inside each folder.

If the star appears significantly off-center when apertures are applied, try one of the following:
Set MANUAL_FOR_ALL_SHOTS = True , to manually click the center for each frame.
or
USE_SEP = True, to enable automatic source detection.

Then rerun proc_RBPL.py.

⸻

Step 5 — Register processed stars

After selecting apertures for all stars, add the names of the star folders to: coursework/0_raw_data/R/uniq_st_names.txt
Example:
Mark_24
Mark_25
Mark_26
Mark_26B

⸻

Step 6 — Collect raw polarization measurements

Navigate to: coursework/0_raw_data/R/  and run:
conda activate mark
python collect_Mark.py

This produces, for each star folder:
 • q_u.png
 • x_y.png
 • Mark_#_raw.dat
and also generates a combined file: Markkanen.dat

⸻

Step 7 — Remove outliers if necessary

Inspect the file q_u.png inside each star folder.
If outliers are present, comment the corresponding entries in:

apertures.txt  using #, and rerun:  python collect_Mark.py

If collect_Mark.py fails for a particular star folder, first check whether something is missing or incorrectly formatted in apertures.txt.

⸻

Step 8 — Apply instrumental correction

Run: python coursework/1_correct_instr/correct_instr_pol.py
This produces the corrected file:

Markkanen_final.dat

which is saved both in:  coursework/1_correct_instr/ and  coursework/0_data/R/

⸻

Step 9 — Merge repeated observations

Navigate to:  coursework/0_data/R/

and run:  python merge_markkanen.py Markkanen_final.dat

This creates the final merged dataset:

Markkanen_final_merged.dat


⸻

Final outputs

Main final data products:

Raw measurements per star
Mark_#_raw.dat

Combined raw dataset
Markkanen.dat

Instrument-corrected data
Markkanen_final.dat

Final merged dataset
Markkanen_final_merged.dat
