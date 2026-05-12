import pandas as pd
from astroquery.utils.tap.core import TapPlus

WHICH = "robopol"   # change to "panopoulou" for second run

if WHICH == "robopol":
    ids_file = "zgr23_query_robopol.csv"
    out_file = "zgr23_robopol.csv"
else:
    ids_file = "panopoulou_ids.csv"
    out_file = "zgr23_panopoulou.csv"

tap = TapPlus(url="https://dc.g-vo.org/tap")

query = """
SELECT m.source_id, x.ext, x.err_ext, x.quality_flags,
       x.mod_parallax, x.err_mod_parallax
FROM TAP_UPLOAD.my_stars AS m
JOIN xpparams.main AS x ON m.source_id = x.source_id
"""

job = tap.launch_job_async(
    query=query,
    upload_resource=ids_file,
    upload_table_name="my_stars",
)
result = job.get_results().to_pandas()
print(f"{len(result)} rows returned")
result.to_csv(out_file, index=False)
print(f"saved -> {out_file}")
