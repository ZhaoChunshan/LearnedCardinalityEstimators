# DACE Checkpoints

- `DACE_author_pretrained.ckpt`: Pretrained weights released by the DACE authors in their [original repository](https://github.com/liang-zibo/DACE/tree/main/checkpoints).
- `DACE_ours_no_imdb.ckpt`: We trained these weights following the steps in the [original README](../README_original.md), excluding IMDb from the training data.
- `feature_statistics.json`: Feature normalization statistics and the node-type encoding used with `DACE_ours_no_imdb.ckpt`. Preserved from the original experiment's `data/workload1/statistics.json`; the node-type order must match the trained model.
