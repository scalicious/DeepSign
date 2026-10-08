# AquaSign: Development Notes & Risk Tracker

## Decisions & Deviations Log
1. **Decision (Phase 0): Domain Scenario Mapping**
   - Verified against Chavez et al. (JMSE 2019):
     - Sea domain: `biograd-A`, `biograd-B`, `biograd-C` (4,467 retained pairs).
     - Pool domain: `brodarski-A`, `brodarski-B`, `brodarski-C`, `brodarski-D`, `genova-A` (4,724 retained pairs).
2. **Decision (Phase 0): Filtered Classes**
   - Out of 16 original gesture classes, 15 survive the `min_per_domain: 40` threshold in both domains:
     `backwards`, `boat`, `carry`, `down`, `end_comm`, `four`, `here`, `mosaic`, `num_delimiter`, `one`, `photo`, `start_comm`, `three`, `two`, `up`.
   - Dropped class: `five` (total 48 samples; under 40 in sea domain).
3. **Decision: Stereo Pairs**
   - Left image is exclusively used for model input as specified. Right image is logged in `index.csv` for stereo pair ID tracking.
4. **Decision: AdaBN Scope**
   - AdaBN recomputes running mean and variance on all `BatchNorm2d` layers in the backbone at test-time without parameter updates or label access.

---

## Known Risks & Tracking (Spec Section 15)
1. **Scenario-class overlap:** Checked in Phase 0. 15 classes retained (>8), 4,724 pool and 4,467 sea pairs (>500 each). Experiment is valid.
2. **Residual leakage:** Mitigated by contiguous block partitioning of 50 samples by temporal `order_key`.
3. **Hand size at 144x192:** Aspect ratio is preserved (640x480 is 4:3, 144x192 is 4:3).
4. **Compute:** AMP + cached uint8 `.npy` ensures fast epoch execution on T4 GPU (<10 min per run).
5. **Correction module efficacy:** Will be monitored via identity initialization and visual inspection.
6. **Confidence head vs MSP:** Selective prediction evaluation compares AURC and selective accuracy against MSP baseline honestly.
7. **Synthetic haze proxy:** Explicitly treated as a turbidity proxy, not true ocean optical physics.
8. **License:** Verified CC BY 4.0.
