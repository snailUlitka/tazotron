Pipeline overview (rough sketch):

### Synthetic CT generation
- Input: healthy CT volume.
- If coarse pelvic segmentation is available (see `DeepFluoroLabeling-IPCAI2020` datasets), locate femoral hemisphere inside the thigh segment and inject necrosis.
- If fine segmentation is available, inject necrosis directly into the acetabular cup segment.
- If no segmentation is available, locate the hemisphere heuristically and inject necrosis.

### Required components
- Synthetic necrosis generator: CT → CT with simulated necrosis.
- CT-to-radiograph renderer: single CT → batch of DRRs / X-rays.
- CT segmenter: produces masks when labels are missing (input CT → segmentation masks).
- Rule-based sphere/hemisphere detector for anatomical localization.
- Binary classifier: X-ray → {necrosis present, absent}.

Reference dataset info (sphere centers, etc.): https://github.com/rg2/DeepFluoroLabeling-IPCAI2020?tab=readme-ov-file#datasets
