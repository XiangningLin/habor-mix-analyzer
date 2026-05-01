# Benchmark info coverage (72 adapters)

All templates live in **`benchmark_info_jobs/`**: **one JSON file per adapter**, filename = slug (see [`benchmark_assignments_ordered.md`](benchmark_assignments_ordered.md)).

Schema / field guide: **`benchmark_info_template.md`**.

## Source list

Slugs match the current **adapter registry** you provided (including items such as OfficeQA, CyberGym, browsecomp, SWE-Bench-Live, etc.). **GAIA** and **GAIA2** are separate rows.

## Notes

- **`researchcodebench.json`**: registry string was misspelled `reaserchcodebench`; file uses the canonical stem `researchcodebench`.
- **`swe_fficiency.json`**: registry shows **SWE-fficiency**; confirm the official benchmark spelling when filling links.

## What to do

1. Open [`benchmark_assignments_ordered.md`](benchmark_assignments_ordered.md), find your adapter, edit the listed file under `benchmark_info_jobs/`.
2. Pull facts from official sites / papers / repos / leaderboards only.
3. Fill every JSON field per the template.

There is **no job / remainder split** in this revision—only this flat list of **72** adapters.
