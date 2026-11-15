# Notebooks

Both run on Colab and neither is needed to run the claim system in `../`. Their
executed outputs are saved, so the figures quoted in the main README can be checked
by opening them rather than re-running anything.

| notebook | what it is |
| --- | --- |
| `train_detectors.ipynb` | trains the two detectors: the 2x2 ablation, the seven runs, and the per-image metrics that chose the deployed model |
| `vlm_comparison.ipynb` | the second vision approach: SmolVLM zero-shot on the same held-out splits |

`train_detectors.ipynb` is where the deployed weights come from. `final_r50_640_40ep`
is the damage model and `parts_frcnn_640` is the parts model; both `best.pt` files go
in `../weights/` as `damage.pt` and `parts.pt`.

It is also where these numbers in the main README come from:

- located 0.86 and right type 0.86 at threshold 0.6, on 81 held-out damage images
- 78% of the 777 labelled damage objects missed, 18% found and named correctly
- mAP@50 0.130 damage, 0.830 parts
- Scratch 8%, Paint chip 0%, Other damage 0% per-class recall

`vlm_comparison.ipynb` is standalone: it rebuilds the test splits from each training
run's own `splits.json` on Drive, so both vision approaches are scored on the
identical images.
