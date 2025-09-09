

Here we create the instrument model


1. Edit corr_QUxy_plane_unpol.py and add there all stars that you have the raw data for.

2. Make sure all the stars are also listed in stand_pol.py with their q, u parameters
   they can be found in table 2 of https://arxiv.org/pdf/2307.06151

3. under 'mark' environment run
   python corr_QUxy_plane_unpol.py
   it will create 4 plot to control the fitting results
   and fit_Q2024.pickle, fit_U2024.pickle that contain the instrumental polarization model parameters
   these files will be later used by the instrumental model correction script

4. edit (if needed) and run python find_inst_rotation_together.py
   it will produce
      EVPA_zp.txt
      PA_diff.pdf

