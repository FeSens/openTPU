---
name: post-route-timing-check
description: Check post-route static timing (TNS/WNS) in OpenROAD to decide whether a macro placement exploration iteration is done. Use this skill whenever the user is running macro placement exploration, sweeping floorplans, or asking whether a placement run is "good enough to stop" — anytime post-route timing in OpenROAD needs to be read and compared against a goal, even if the user doesn't explicitly say "TNS" or "WNS".
---

# Post-Route Timing Check (OpenROAD)

Used inside a macro placement exploration loop. After routing completes in OpenROAD, this skill reads the design's timing health and decides whether the loop should stop or continue with another placement candidate.

## When to run

Run this skill **only after detailed routing has finished** for a placement candidate. Pre-route or post-CTS slack is based on estimated parasitics and will mislead the exploration loop — drop those numbers, don't act on them.

## Commands

In the OpenROAD shell (or `.tcl` script), run both, in this order:

```tcl
report_tns
report_wns
```

- **WNS** (Worst Negative Slack) — the single worst path. Tells you if any path is failing.
- **TNS** (Total Negative Slack) — the sum of negative slack across all failing endpoints. Tells you how much failure there is in aggregate.

Both are reported in the design's time unit (typically nanoseconds). A clean design reports `WNS = 0` and `TNS = 0`. Negative values indicate failing paths.

## Stop criteria

Stop the macro placement exploration when **either** condition is true:

1. **Timing goal met** — `WNS >= 0` AND `TNS >= 0` (fully clean timing, no negative slack anywhere).
2. **Try limit exhausted** — the exploration has used its budget of placement attempts.

If neither condition is true, return control to the placement loop to try the next candidate.

Why both conditions matter: without the try limit, a design that cannot close timing under any placement will loop forever. Without the goal check, you'll burn the full budget even on a placement that already closes on iteration 2.

## Reporting

After each check, report back in this exact format so the loop driver can parse it:

```
Iteration: <N> / <try_limit>
WNS: <value> ns   (goal: >= 0)
TNS: <value> ns   (goal: >= 0)
Decision: STOP (goal met) | STOP (try limit exhausted) | CONTINUE
```

Keep the output terse — this skill runs inside a loop and verbose output clutters the exploration log.
