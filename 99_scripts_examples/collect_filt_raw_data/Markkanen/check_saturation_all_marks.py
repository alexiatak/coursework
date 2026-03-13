
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.ndimage import maximum_filter


ROOT_DIR = Path("./Markkanen")         # folder where all Mark_* folders are
SAT_LIMIT = 50000
MIN_PEAK = 1000              # ignore weak noise peaks
EXCLUSION_RADIUS = 20        # so one spot is not selected many times
BOX_HALFSIZE = 6             # local box around each bright spot



def find_four_spot_maxima(
    data,
    min_peak=MIN_PEAK,
    exclusion_radius=EXCLUSION_RADIUS,
    box_halfsize=BOX_HALFSIZE,
):
    
    #Find the 4 brightest separated peaks and return the local maxima around them.
   
    mx = maximum_filter(data, size=7)
    peaks = (data == mx) & (data > min_peak)

    ys, xs = np.where(peaks)
    if len(xs) == 0:
        return []

    vals = data[ys, xs]
    order = np.argsort(vals)[::-1]

    selected = []
    for idx in order:
        y, x = ys[idx], xs[idx]

        too_close = False
        for sy, sx in selected:
            if (y - sy) ** 2 + (x - sx) ** 2 < exclusion_radius ** 2:
                too_close = True
                break

        if not too_close:
            selected.append((y, x))

        if len(selected) == 4:
            break

    maxima = []
    for y, x in selected:
        y1 = max(0, y - box_halfsize)
        y2 = min(data.shape[0], y + box_halfsize + 1)
        x1 = max(0, x - box_halfsize)
        x2 = min(data.shape[1], x + box_halfsize + 1)
        local_max = np.max(data[y1:y2, x1:x2])
        maxima.append(local_max)

    return maxima


def check_fits_file(fits_path):
    """
    Returns:
        maxima: list of maxima for the 4 spots
        saturated: True/False
        status: text status
    """
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data.astype(float)
            data = np.squeeze(data)

        if data.ndim != 2:
            return [], False, "NOT_2D"

        maxima = find_four_spot_maxima(data)

        if len(maxima) < 4:
            return maxima, False, "FAILED_TO_FIND_4_SPOTS"

        saturated = any(m > SAT_LIMIT for m in maxima)
        return maxima, saturated, "OK"

    except Exception as e:
        return [], False, f"ERROR: {e}"



def main():
    mark_dirs = sorted(
        [p for p in ROOT_DIR.iterdir() if p.is_dir() and p.name.startswith("Mark_")]
    )

    if not mark_dirs:
        print("No Mark_* folders found.")
        return

    found_any_saturated = False

    for mark_dir in mark_dirs:
        inner_24_dirs = sorted(
            [p for p in mark_dir.iterdir() if p.is_dir() and p.name.startswith("24")]
        )

        if not inner_24_dirs:
            continue

        for inner_dir in inner_24_dirs:
            fits_files = sorted(inner_dir.glob("*.fits"))

            for fits_file in fits_files:
                maxima, saturated, status = check_fits_file(fits_file)

                if status != "OK":
                    print(f"[{status}] {mark_dir.name} -> {fits_file}")
                    continue

                if saturated:
                    found_any_saturated = True
                    print(f"[SATURATED] {mark_dir.name} -> {fits_file.name}")
                    print(f"            maxima = {[round(m, 1) for m in maxima]}")

    if not found_any_saturated:
        print("No saturated FITS files found.")


if __name__ == "__main__":
    main()
