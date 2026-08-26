# Reply to NVIDIA Customer Care (Aadya) — DGX Spark thermal case

Hi Aadya, thank you for the quick response. Answers to every item below, with
measured telemetry. Summary up front: the unit hard powers off (instant, no
shutdown sequence, nothing in logs) when platform thermal zones reach ~91°C,
which occurs within ~45 seconds of sustained GPU load at under 100 W. The GPU
die sensor stays 12–15°C BELOW the platform zones throughout, which points to
degraded heat transfer (TIM/heatsink contact) rather than fan or ambient.

**1. Does the system feel unusually hot, throttle, freeze, restart, or shut down?**
It hard shuts down (instant power-off, no OS shutdown sequence, system journal
cut mid-line). Four occurrences: Aug 22 08:02, Aug 26 02:15, Aug 26 12:05,
Aug 26 15:48 (all logged in journalctl boot history). Before shutdown the BMC
appears to cap GPU power at ~60–95 W despite 95%+ utilization, with no kernel
throttle-reason flag set.

**2. GPU temperature at idle and under the workload**
- Idle (measured just now): GPU die 44°C at 11.7 W, ACPI platform zones 44–47°C.
- Under load: GPU die 64–83°C — but ACPI platform zones (acpitz) climb from
  ~80°C to 89–91°C within 45 seconds at 92–97 W. The 12–15°C die-vs-platform
  gap persists throughout.

**3. Screenshot / nvidia-smi output**
Attached: (a) controlled load test log — timestamps, per-zone temps, nvidia-smi
power/temp/util every ~11 s until safety abort at 89.4°C; (b) a black-box
telemetry capture (30-second fsync'd samples) of the actual Aug 26 15:48
shutdown: last recorded sample shows 96% util, 67 W, zones at 91.2°C, followed
by instant power loss. [attach results/thermal_verify.log and the final section
of results/blackbox.log; screenshots of nvidia-smi can be provided on request]

**4. Workload running when the issue occurs**
PyTorch 2.13 (cu130) language-model training, native (no container), batch
size 16–32, ~120M-parameter model. Also reproduced with a plain PyTorch
matmul loop — the issue is load-dependent, not application-specific.

**5. Time for the problem to appear**
Thermal runaway begins immediately under load: platform zones go 79.6°C →
89.4°C in 44 seconds at ≤97 W. Hard shutdowns occurred between ~2 minutes and
~4.3 hours into training runs (duty-cycling workloads last longer). Recovery
is equally fast when load stops (85°C → 68°C within 10–20 s), consistent with
a localized heat-transfer defect.

**6. Ambient room temperature**
[OPERATOR: fill in, e.g. "~22°C, air-conditioned room"] — well within the
5–30°C range. Note: an RTX 5090 workstation in the same room sustains 483 W
at 69°C GPU, so ambient is demonstrably not the cause.

**7. Ventilation / vents unobstructed?**
[OPERATOR: confirm, e.g. "Yes — unit is on an open desk, vents clear, nothing
stacked on or around it."]

**8. Fan spinning / unusual noise?**
[OPERATOR: confirm what you hear under load.] No fan RPM sensor is exposed to
the OS (no hwmon fan inputs), so we cannot verify in software. The very fast
cool-down when load stops suggests airflow exists; the defect pattern matches
heatsink/TIM contact rather than a stopped fan.

**9. Supplied NVIDIA 240 W adapter in use?**
[OPERATOR: confirm "Yes, original NVIDIA adapter and cable."]

**10. DGX OS, driver, firmware versions**
- DGX OS: 7.2.3 (build 2025-09-10, commit 833b4a7), Ubuntu 24.04.4 LTS
- Kernel: 6.17.0-1031-nvidia (issue also reproduced on 6.17.0-1029-nvidia)
- GPU driver: 580.173.02 (open kernel module)
- UEFI/system firmware: 0x03000508; UEFI PK 2025; NVMe FW NXHB202Q
- fwupdmgr reports Embedded Controller and UEFI at latest, no pending updates

**11. All DGX Spark updates installed?**
Yes — fwupdmgr shows no available firmware updates; OS kernel is current
(6.17.0-1031-nvidia, installed Aug 26). A handful of routine apt package
updates are pending but none are firmware/driver/thermal related.

**12. Connected devices**
Headless operation: no display, no external storage. Only internal WiFi/BT
module on USB; gigabit LAN connected. Nothing else attached.

**13. Opened, modified, or physically damaged?**
No. The unit has never been opened or modified and has no physical damage.

**14. Screenshots of thermal warnings/errors**
There are none — that is the core symptom. The shutdown is instantaneous with
no OS-level warning, message, or shutdown sequence; the system journal simply
stops mid-line (timestamps above). Our own fsync'd telemetry recorder captured
the final seconds (attached).

**15. Troubleshooting already completed**
- Ruled out: memory pressure (sar shows ~20% RAM used at every event), disk
  (1.3 TB free), kernel version (reproduced on 1029 and 1031), firmware level
  (EC/UEFI current), OOM/panic (none logged), ambient (see #6).
- Controlled reproduction with safety abort: documented 10°C/30 s runaway at
  <100 W with a persistent 12–15°C die-vs-platform gap.
- Interim mitigation in place: a software governor pauses workloads above 85°C
  (SIGSTOP/SIGCONT), which prevents shutdowns but caps the machine at a
  fraction of its rated capability.
- **Field Diagnostics**: dgx-spark-fieldiag is installed; `partnerdiag --field`
  fails at MODS driver insertion because Secure Boot is enabled ("Key was
  rejected by service") and this unit runs headless (no console available to
  complete a MokManager/Secure Boot change remotely). We can arrange one-time
  physical access to disable Secure Boot and run FieldDiag if required —
  please confirm whether you need it to proceed, given the telemetry above.

Serial number: [OPERATOR: output of `sudo dmidecode -s system-serial-number`]

We believe this matches the known GB10 thermal-defect pattern discussed in
developer-forum threads 349647, 363370, 373266 and 380238 (the latter two
resulting in RMAs). Requesting RMA evaluation.
