## Included files

- `python/eval_bert_softmax_tasks.py`
  Task-level evaluator that replaces attention softmax during inference.
- `python/softmax_edge_model.py`
  Core approximate softmax model used by the evaluator.
- `python/run_glue_full_offline.py`
  GLUE entrypoint.
- `python/run_squad_offline.py`
  SQuAD entrypoint.
- `third_party/metrics/`
  Local metric scripts used by `evaluate`.

## Default settings

- `exact`
- `doc_adaptive_desc9_q7_special4` for `block_size=8`

## Run GLUE

```bash
python model/python/run_glue_full_offline.py --max-samples 0 --output-json model/results/glue.json
```

## Run SQuAD

```bash
python model/python/run_squad_offline.py --tasks squad --max-samples 0 --output-json model/results/squad.json
```

## Useful options

```bash
python model/python/run_glue_full_offline.py --block-size 4
python model/python/run_squad_offline.py --profiles exact,doc_adaptive_desc9_q7_special4_block4
```

## Notes

- The evaluator replaces attention softmax, not the final classifier head.
- CPU execution is supported by default.
- The first run may take a long time due to model and dataset downloads.

