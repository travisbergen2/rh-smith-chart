# E-LI-1b — Gate Outcomes and Endpoint Results (2026-07-15)
(To append to protocol doc cmrmpxh7i01bx07ad73p7145m, section 8, when doc tools return.)

Pipeline: /agent/workspace/e_li_1b/. Zeros: Odlyzko zeros6, first 2,001,052 (T~1,132,490.66).

G1 PASS both arms (Li 1.5e-8 vs 1e-6 gate; Paper 14 mu_min table to displayed digits;
two documented accuracy repairs: lambda tail log-singularity, ARCH rebuild stable 1e-11).
G2 PASS: binding D_max = 2000 (Li 3.7e-8; Toeplitz max 0.16%).
N0 PASS (s_n min 6.7e-2; t_M min 9.8e-8).

Endpoint (12 cells, seeds 6000-6019, mass-matched nulls):
gamma*   eps   N_li    N_toeplitz   mu_min(D_max)
14.1347  0.10  1244    87           -1.0e5
14.1347  0.05  --(-1.81s) 162       -1.3e3
14.1347  0.02  --      272          -77
14.1347  0.01  --      387          -16
98.8312  0.10  --      272          -6.9e3   <- matches E-LI-1 exactly
98.8312  0.05  --      437          -87
98.8312  0.02  --      817          -5.0
98.8312  0.01  --      1167         -1.0
999.7916 0.10  --      882          -3.3
999.7916 0.05  --      1257         -0.037
999.7916 0.02  --      1837         -7.6e-4
999.7916 0.01  --      --           +1.6e-7 UNRESOLVED

R = 11/12 resolved >= 7 (powered). G3a PASS (1e-11); G3b substitute PASS (0 flips,
sign-off pending); G3c: all detections by genuine indefiniteness (sign-only = primary
depth in all 11 cells), normalization-invariant; G4 PASS (5.7e-14, signs agree).
Jitter spot-check: 0.016% of band SD (bound 10%).

Theta arm: adequacy PASS; window spot-check 1.5e-5. MECHANISM FINDING: registered
eps > omega staircase wrong in 18/72 (all misses, no false alarms); measured rule
eps - omega >= v (pole must reach probe line) matches 72/72. Gauge is local in x AND v.

Power table vs measured: Toeplitz in/better than predicted bands everywhere; two
predicted-UNRESOLVED cells resolved (98.83/0.01 at 1167; 999.79/0.02 at 1837);
no 3x deviation. Li prediction miss at (14.13, 0.05): -1.81 sigma vs -3 bar.

REGISTERED VERDICT (per frozen section 5):
R = 11 >= 7. Per-cell winners: (14.13, 0.1) both resolve, 87 <= 1244/2 -> TOEPLITZ;
10 sole-resolver cells -> TOEPLITZ. Toeplitz wins 11/11 resolved (100% >= 70%),
loses 0 (<= 10%).
VERDICT: TOEPLITZ WINS.
Caveats bound to verdict: G3(b) substitute awaits Travis sign-off (striking it would
not change the verdict: all Toeplitz detections are sign-based and normalization-
invariant; G3b touches only the Li arm's single detection). Depth grid step 5 and
windowed jitter per Deviations Entries 2/4 with passed spot-checks.
