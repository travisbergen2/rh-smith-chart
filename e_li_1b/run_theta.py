"""E-LI-1b Theta_omega secondary arm + Deviations spot-checks.

Registered probe: x in [-g*-3, -g*+3], dx=0.005, v in {0.02, 0.05},
omega in {0.005, 0.02, 0.05}. Detection: peak > 1 + 5*SD(null peaks).
G5 adequacy: null peaks within [0.95, 1.05] else arm DEGRADED.
Window |gamma - gamma*| <= 500 (Deviations entry 4a) with one
full-product spot-check.
"""
import json
import numpy as np
import lib1b

G = lib1b.load_zeros()
targets = {}
for gs in lib1b.GAMMA_TARGETS:
    idx = int(np.argmin(np.abs(G - gs)))
    targets[gs] = float(G[idx])

results, adequacy = [], []
for gs, gstar in targets.items():
    win_mask = np.abs(G - gstar) <= lib1b.THETA_WINDOW
    G_win = G[win_mask]
    # head part of the window is jittered per seed (all window zeros are
    # <= 5000 for our targets, so the whole window jitters)
    for omega in lib1b.OMEGAS:
        for v in lib1b.PROBE_VS:
            null_peaks = []
            for seed in lib1b.SEEDS_CONF:
                rng_head = lib1b.jittered_head(G_win, seed)  # window jitter
                i_near = int(np.argmin(np.abs(rng_head - gstar)))
                g_dup = float(rng_head[i_near])
                pk = lib1b.theta_peak(rng_head, omega, gstar, v,
                                      extra_online=[g_dup])
                null_peaks.append(pk)
            null_peaks = np.array(null_peaks)
            adequacy.append({"gamma_star": gstar, "omega": omega, "v": v,
                             "null_min": float(null_peaks.min()),
                             "null_max": float(null_peaks.max())})
            sd = float(null_peaks.std(ddof=1))
            for eps in lib1b.EPSILONS:
                pk_mod = lib1b.theta_peak(G_win, omega, gstar, v,
                                          quad_eps=eps, drop_nearest=gstar)
                fire = pk_mod > 1.0 + 5.0 * sd
                results.append({
                    "gamma_star": gstar, "eps": eps, "omega": omega, "v": v,
                    "peak_mod": float(pk_mod), "null_sd": sd,
                    "detect": bool(fire),
                    "predicted": bool(eps > omega)})
            print(f"theta g*={gstar:.2f} om={omega} v={v} done", flush=True)

# Deviations entry 4a spot-check: full product vs windowed, one config
gstar = targets[98.831194]
pk_win = lib1b.theta_peak(G[np.abs(G - gstar) <= lib1b.THETA_WINDOW],
                          0.05, gstar, 0.05, quad_eps=0.1, drop_nearest=gstar)
pk_full = lib1b.theta_peak(G, 0.05, gstar, 0.05, quad_eps=0.1,
                           drop_nearest=gstar)
spot = {"windowed": float(pk_win), "full": float(pk_full),
        "rel_diff": float(abs(pk_full - pk_win) / pk_full)}

staircase_ok = all(r["detect"] == r["predicted"] for r in results)
out = {"results": results, "adequacy": adequacy,
       "adequacy_pass": all(0.95 <= a["null_min"] and a["null_max"] <= 1.05
                            for a in adequacy),
       "window_spotcheck": spot,
       "staircase_matches_prediction": staircase_ok}
with open("data/theta_report.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1))
print("mismatches:", [r for r in results if r["detect"] != r["predicted"]])
