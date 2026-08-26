# DGX Spark (GB10): hard power-off at ~91°C platform temp under sustained load — telemetry attached, FieldDiag blocked by Secure Boot on headless unit

**PRIMARY (official) channel — NVIDIA Customer Care ticket:**
http://nvidia.custhelp.com/app/ask  (or Live Chat: http://nvidia.custhelp.com/app/chat/chat_launch/)
Per the DGX Spark support docs (docs.nvidia.com/dgx/dgx-spark/support.html),
this is the hardware-support path; FieldDiag is an RMA *pre-check*, not a
prerequisite for opening the case. Warranty: 1 year (new units).

**Secondary (optional, parallel): community forum**
https://forums.developer.nvidia.com/c/dgx/dgx-spark/
Related threads: 349647 (thermal throttling), 363370 (auto shutdown),
373266 (FieldDiag PowerStress MODS-020000600139 → RMA), 380238 (silent
hard-locks → RMA'd) — establishes this as a known defect pattern.

---

## Summary

Our DGX Spark hard powers off (instant, no shutdown sequence, nothing in the
journal) whenever platform thermal zones reach ~91°C, which happens within
**~45 seconds** of sustained GPU load at only **~95 W**. GPU die temperature
stays 12–15°C *below* the platform zones throughout (die 67–77°C while
platform hits 89–91°C), pointing at a degraded heat-transfer path
(TIM/heatsink contact) rather than fan curves. Reproduced across kernels
6.17.0-1029-nvidia and 6.17.0-1031-nvidia; EC and UEFI are at latest
(fwupdmgr reports no updates).

- Product: DGX Spark (GB10), hostname spark-5208
- Serial: (run `sudo dmidecode -s system-serial-number`)
- Driver: 580.173.02 (open kernel module) · Kernel: 6.17.0-1031-nvidia
- Firmware: EC + UEFI current per fwupdmgr (no pending updates)

## Crash history (journalctl --list-boots + last -x)

Four hard stops, each with the journal cut mid-line, no shutdown sequence,
no kernel panic, no OOM (memory ~20% used at every event per sysstat/sar):

| When | Context |
|---|---|
| Aug 22 08:02 | ~2 min after a 110M-param training run resumed |
| Aug 26 02:15 | low load |
| Aug 26 12:05 | ~4.3 h into training (new 1031 kernel) |
| Aug 26 15:48 | training; **captured by our fsync'd telemetry recorder (below)** |

## Black-box capture of the Aug 26 15:48 power-off

30-second samples, fsync'd, last records before the cut:

```
15:47:24  zones(mC): 90300 78400 81000 79500 88000 90300 81500 | GPU 69.4W 83C 95%
15:47:54  zones(mC): 90300 78700 79100 79700 90300 89900 81500 | GPU 63.1W 80C 96%
15:48:24  zones(mC): 91200 79300 82200 80300 89800 91200 81800 | GPU 67.3W 82C 96%
<instant power-off; journal's final line is a routine cron entry at 15:45:01>
```

## Controlled reproduction (safety-aborted before the trip)

Sustained matmul load, thermal zones sampled continuously, abort at 88°C:

```
 11s  maxzone=79.6C  gpu=[97.3W, die 67C, 96%]
 22s  maxzone=83.8C  gpu=[94.4W, die 64C, 82%]
 33s  maxzone=86.8C  gpu=[94.9W, die 75C, 82%]
 44s  maxzone=89.4C  gpu=[92.2W, die 77C, 96%]
SAFETY ABORT — verdict: thermal runaway, ~10°C/30s at ≤97W
```

Note the persistent 12–15°C gap between GPU die and platform zones at
under 100 W. For comparison, an RTX 5090 in the same room sustains 483 W
at 69°C — ambient is not a factor.

Recovery is equally fast (85→68°C in ~10–20 s once load stops), consistent
with a localized heat-transfer defect rather than airflow limitation.

## FieldDiag status

`dgx-spark-fieldiag` installed; `partnerdiag --field` fails at MODS driver
insertion under Secure Boot: `insmod: ERROR: could not insert module
mods.ko: Key was rejected by service`. The unit is headless (SSH only);
`/var/lib/shim-signed/mok/` is empty on this image so there is no local
MOK to sign with, and MOK enrollment/Secure Boot changes require a physical
console we can attach only with difficulty. Happy to run FieldDiag after a
physical-console session if required — but the PowerStress outcome is
predictable from the reproduction above (expect MODS-020000600139 per
thread 373266).

## Request

This matches the known GB10 thermal defect pattern in the threads above.
Requesting RMA evaluation. Full logs available: black-box telemetry,
thermal-governor pause/resume log, journalctl boot list, sar memory
records, controlled-reproduction log.
