# Dev labels

Each person writes `labels/raw/<name>.csv` (UTF-8, comma-separated, header line required):

```
video,start,end,label,notes
C3902.MP4,12.4,17.9,jaywalking,woman crossing mid-block
C3902.MP4,1:03.5,1:09,red_light,"white van, lane 2"
C3905.MP4,,,none,watched fully, nothing happened
```

- `video`: the file name exactly as in `samples/`.
- `start`, `end`: seconds (`75.5`) or `m:ss.s` (`1:15.5`). Use the start/end rules of `docs/SPEC.md` §5
  (for example red_light starts when the front crosses the stop line, not when the car appears).
- `label`: one of the 14 ids in `solution.CLASSES`. Write `none` (no times) for a video you watched
  fully without events, so it still counts as labelled.
- `notes`: anything unsure; P2 decides. Quote the field if it contains a comma.

Then `python scripts/labels_to_gt.py` validates every row and writes `labels/dev_gt.json`
(nothing is written if a row is wrong), and `python scripts/eval_dev.py` scores the pipeline on it.
