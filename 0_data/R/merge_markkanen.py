"""
Merge repeated Markkanen stars in Markkanen_final.dat.

It keeps stars that appear only once unchanged.
It merges stars like:
    Mark_24 + Mark_24B           -> Mark_24_merged
    Mark_26 + Mark_26B + ...     -> Mark_26_merged
    Mark_29 + Mark_29B           -> Mark_29_merged

The averaging is done in Stokes q and u.
"""

from pathlib import Path
import re
import numpy as np
from uncertainties import ufloat
from uncertainties import umath
from scipy.special import erf
from scipy.optimize import minimize
import scipy.integrate as integrate

INPUT_FILE = "Markkanen_final.dat"
OUTPUT_FILE = "Markkanen_final_merged.dat"

class Obs:
    def __init__(self, name, JD, q, qerr, u, uerr, x=0.0, y=0.0):
        self.name = name
        self.JD = JD
        self.q = ufloat(q, qerr)
        self.u = ufloat(u, uerr)
        self.x = x
        self.y = y

    def getP(self):
        return umath.sqrt(self.q ** 2 + self.u ** 2)

    def EVPA_pdf(self, theta, P0):
        g = 1 / np.sqrt(np.pi)
        ita0 = float(P0) / np.sqrt(2) * np.cos(2 * theta)
        g = g * (g + ita0 * np.exp(ita0 ** 2) * (1 + erf(ita0)))
        g = g * np.exp(-(float(P0) ** 2) / 2)
        return g

    def int_eq(self, sigma, snr):
        integ = integrate.quad(lambda x: self.EVPA_pdf(x, snr), -sigma, sigma)
        return abs(integ[0] - 0.68268949)

    def getSigma(self, pd, pd_err):
        snr = pd / pd_err
        if snr > 20:
            return 0.5 * 1.0 / float(snr)
        if snr < np.sqrt(2.0):
            pd = 0.0
        else:
            pd = np.sqrt(pd ** 2 - pd_err ** 2)
        snr = pd / pd_err
        res = minimize(
            self.int_eq,
            [np.pi / 50],
            args=(snr,),
            method="Nelder-Mead",
            tol=1e-5,
        )
        if res.status != 0:
            print("Warning: EVPA uncertainty calculation failed for", self.name)
            return np.nan
        return res.x[0]

    def getPA(self):
        PAval = 0.5 * umath.atan2(self.u, self.q)
        PAunc = self.getSigma(self.getP().n, self.getP().s)
        return ufloat(np.degrees(PAval.n), np.degrees(PAunc))


def base_name(name: str) -> str:
    m = re.match(r"^(Mark_\d+)([A-Z]+)?$", name)
    return m.group(1) if m else name


def parse_input_file(infile: Path):
    rows = []
    with open(infile, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            rows.append(
                {
                    "name": parts[0],
                    "JD": float(parts[1]),
                    "P": float(parts[2]),
                    "sP": float(parts[3]),
                    "PA": float(parts[4]),
                    "sPA": float(parts[5]),
                    "q": float(parts[6]),
                    "sq": float(parts[7]),
                    "u": float(parts[8]),
                    "su": float(parts[9]),
                }
            )
    return rows


def merge_group(base: str, group_rows):
    obs_list = [
        Obs(r["name"], r["JD"], r["q"], r["sq"], r["u"], r["su"])
        for r in group_rows
    ]
    n = len(obs_list)

    q_combined = sum(obs.q for obs in obs_list) / n
    u_combined = sum(obs.u for obs in obs_list) / n
    jd_combined = sum(obs.JD for obs in obs_list) / n

    st_combined = Obs(
        f"{base}_merged",
        jd_combined,
        q_combined.n,
        q_combined.s,
        u_combined.n,
        u_combined.s,
    )

    P = st_combined.getP()
    PA = st_combined.getPA()

    return {
        "name": st_combined.name,
        "JD": st_combined.JD,
        "P": P.n * 100.0,    # convert fraction to %
        "sP": P.s * 100.0,
        "PA": PA.n,
        "sPA": PA.s,
        "q": st_combined.q.n,
        "sq": st_combined.q.s,
        "u": st_combined.u.n,
        "su": st_combined.u.s,
        "members": [r["name"] for r in group_rows],
    }


def main():
    
    rows = parse_input_file(INPUT_FILE)

    groups = {}
    for row in rows:
        groups.setdefault(base_name(row["name"]), []).append(row)

    # sort by Mark number
    def keyfunc(item):
        m = re.search(r"(\d+)$", item[0])
        return int(m.group(1)) if m else 10**9

    output_rows = []
    merged_info = []

    for base, group_rows in sorted(groups.items(), key=keyfunc):
        if len(group_rows) == 1 and group_rows[0]["name"] == base:
            row = group_rows[0].copy()
            row["members"] = [row["name"]]
            output_rows.append(row)
        else:
            merged = merge_group(base, group_rows)
            output_rows.append(merged)
            merged_info.append((base, merged["members"]))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("#Name   JD    P[%]   sP[%]    PA[deg]  sPA[deg]   q        sq       u        su\n")
        for row in output_rows:
            f.write(
                f'{row["name"]:<16s} '
                f'{row["JD"]:12.5f} '
                f'{row["P"]:6.2f} '
                f'{row["sP"]:6.2f} '
                f'{row["PA"]:8.2f} '
                f'{row["sPA"]:8.2f} '
                f'{row["q"]: .5f} '
                f'{row["sq"]: .5f} '
                f'{row["u"]: .5f} '
                f'{row["su"]: .5f}\n'
            )

    print(f"Saved: {OUTPUT_FILE}")
    print("Merged groups:")
    for base, members in merged_info:
        print(f"  {base} -> {base}_merged from {', '.join(members)}")


if __name__ == "__main__":
    main()
