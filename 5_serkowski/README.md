# Serkowski Analysis — README

This folder contains the tools for fitting the Serkowski law to multiband polarization data
for the three Mark stars observed in B, G, R, I filters (Mark_26, Mark_76, Mark_81).

The Fortran fitting code (code.for) was provided by Ilin Vladimir Borisovich.

---

## What the Serkowski law is

The Serkowski law describes how interstellar polarization depends on wavelength:

    P(lambda) = Pmax * exp( -K * ln(lambda_max / lambda)^2 )

where K = 1.7 * lambda_max (Wilking+1982 relation). The fit has two free parameters:
Pmax (peak polarization) and lambda_max (wavelength of peak, in microns). For typical
diffuse ISM dust, lambda_max is around 0.55 micron.

To fit this curve you need polarization measurements in at least 3 bands. With only 4 bands
(B, G, R, I) you get 2 degrees of freedom.

---

## Files in this folder

    code.for                  Original Fortran source from Ilin V.B. (free-format, may need
                              -ffree-form flag or reformatting to compile)

    code_fixed.for            Same code reformatted to strict Fortran fixed format (72-char
                              line limit). Use this one to compile.

    fix_fortran.py            Python script that reformats code.for into code_fixed.for.
                              Run this if code.for is updated.

    make_serkowski_inp.py     Reads the four Markkanen_final_*.dat files (one per band) and
                              writes inp.txt in the format the Fortran program expects.

    check_serkowski_data.py   Diagnostic plot: shows P vs lambda for each star before fitting.
                              Run this first to check whether the data looks physically
                              reasonable before feeding it to the Fortran code.

    inp.txt                   Input file for the Fortran program. Each star is a block with
                              header line (name + number of points) followed by rows of
                              lambda, P, sigma_P.

    out1.txt                  Output: one summary line per star with lambda_max, Pmax, and
                              their uncertainties.

    out2.txt                  Output: per-wavelength comparison of observed vs fitted P/Pmax.

---

## Step-by-step workflow

### Step 1 — Check your data first

Before running anything, run the diagnostic:

    python check_serkowski_data.py

This plots P(lambda) for each star. The curve should rise from B to a peak somewhere around
G or R, then fall toward I. 

### Step 2 — Prepare the input file

Edit the BAND_FILES paths at the top of make_serkowski_inp.py to point to your four
corrected and merged .dat files:

    BAND_FILES = {
        "B": "path/to/Markkanen_final_B.dat",
        "G": "path/to/Markkanen_final_G.dat",
        "R": "path/to/Markkanen_final_R.dat",
        "I": "path/to/Markkanen_final_I.dat",
    }

Also check the effective wavelengths in LAMBDA_EFF. The values used here are:

    B = 0.440 micron
    G = 0.530 micron   
    R = 0.640 micron
    I = 0.800 micron

Then run:

    python make_serkowski_inp.py

This writes inp.txt and also creates empty out1.txt and out2.txt (the Fortran program
requires these files to already exist before it runs).

Important note for Mark_26: the R-band value to use is from Mark_26D, not from the
merged R-band file. Mark_26D has JD = 2460829.35684, which matches the JD of the B, G, I
observations, meaning it is the R channel from the same simultaneous multiband exposure.


### Step 3 — Compile the Fortran code

    gfortran -ffixed-form -ffixed-line-length-132 code_fixed.for -o serk

### Step 4 — Run the fit

    ./serk

The program reads inp.txt, prints iteration progress to the terminal, and writes results
to out1.txt and out2.txt.

### Step 5 — Read the results

out1.txt has one line per star:

    StarName   lambda_max +/- sigma   Pmax +/- sigma   chi2/dof

A chi2/dof close to 1 means a good fit. Much larger than 1 means the Serkowski law does
not describe the data well (could be a real deviation, or bad data).

out2.txt has the per-wavelength breakdown: observed P/Pmax, theoretical P/Pmax, and
the residual.

---


## inp.txt format reference

Each star block looks like this:

     StarName   N
      0.440  P1   sP1
      0.530  P2   sP2
      0.640  P3   sP3
      0.800  P4   sP4

- Star name is left-justified in 8 characters (column 2-9)
- N is the number of wavelength points (integer, 4 digits)
- Each data row: lambda (f6.3), P in percent (f6.3), sigma_P in percent (f7.3)
- The format is strict Fortran fixed format — spacing matters

The Fortran program uses the value at the longest wavelength as the initial guess for
Pmax, and the corresponding lambda as the initial guess for lambda_max. If the I-band
value is the highest, the initial guess is already at the edge of the data range and
the fit will extrapolate to unphysical lambda_max values.
