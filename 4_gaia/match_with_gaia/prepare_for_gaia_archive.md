# prepare_for_gaia_archive.py

Builds the upload CSV for the Gaia archive in a specific format.

## Inputs

- `merged_with_gaiaid.csv` — output of `match_gaia_ids.py`

## What it does

1. Reads the matched table.
2. Drops anything that did not get a Gaia ID.
3. Selects Name, ra, dec, gaia_id and renames `gaia_id` to `gid`.
4. Writes the result.

## Output

`for_gaia_archive.csv` — columns: Name, ra, dec, gid.

## How to use it

1. Go to https://gea.esac.esa.int/archive/.
2. Upload `for_gaia_archive.csv` as a user table called `mark`.
3. Run this ADQL:

   ```sql
   SELECT mark.Name, mark.ra, mark.dec, mark.gid,
          edr3.r_med_photogeo, edr3.r_lo_photogeo, edr3.r_hi_photogeo
   FROM   external.gaiaedr3_distance AS edr3
   JOIN   user_<your_username>.mark   AS mark
     ON   mark.gid = edr3.source_id
   ```

4. Download in /4_gaia the result as `bj_distances.csv` and feed it into the next script.

## Notes


- `Name` is included in the SELECT clause so the result 
can be rejoined to the polarization data without going back through Gaia IDs.
