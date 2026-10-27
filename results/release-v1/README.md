# Release verification

The final local suite passed all 72 tests with no skips, including live PostgreSQL,
Kafka, Chromium and local MLflow. The 217 warnings are upstream deprecations.
A new Python 3.13 environment without system site packages installed the wheel,
served its packaged web assets and loaded the released models on CPU.
The authored model example exactly matched its expected four scores.

- [Package and clean-install evidence](package.json)
- [Actual 93-case development workbench verification](browser.json): both rankings,
  all 14 cold cases, hidden outcomes, export and desktop/mobile layout.
- [Local MLflow verification](tracking.json): 32 metrics, 14 model/package files
  and the exact final report read back successfully.
- [Original model file checksums](model-checksums.json)
- [Final independent audit](../heldout-v1/audit.json)

The model ZIP has 14 files and contains no transaction tables. The included
public workbench example is authored data, not a model evaluation.
Private screenshots use the real IBM development export; no final test cases
were selected for the demo. The initial screenshot/export preparation had
punctuation encoding issues and an asynchronous UI count measurement; it is
retained privately and superseded by the corrected browser evidence here.
No model or score changed.

The earlier exact-SHA service CI and all original experiment reports remain
available. The final workflow additionally runs the browser and tracking tests.
No final training or scoring job is left active. Project containers are stopped
with volumes retained; all four course containers are unchanged.
