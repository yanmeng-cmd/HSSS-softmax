# HSSS Softmax

This repository contains the RTL implementation and model-level evaluation
scripts for the HSSS approximate softmax design.

## Directory Layout

```text
rtl/        Verilog RTL for HSSS Softmax. Vivado 2022.2.
testbench/  Basic RTL testbench.
model/      Python model-level evaluation scripts.
```

The `model/` directory contains:

```text
python/                 BERT GLUE/SQuAD evaluation scripts.
longformer_wiki/        Longformer WikiText-2 MLM evaluation.
longformer_imdb/        Longformer IMDB classification evaluation.
softmax_edge_model.py   Python reference model aligned with RTL.
RESULTS_SUMMARY.md      Key Longformer evaluation results.
```

## Usage

Create the Python environment:

```bash
cd model
make bert env
```

Run BERT evaluations:

```bash
make bert glue
make bert squad
```

Run the BERT block-4 comparison:

```bash
make bert glue PROFILE=HSSS-Softmax-block4
make bert squad PROFILE=HSSS-Softmax-block4
```

Create the Longformer environments and download the required assets:

```bash
make longformer imdb env
make longformer wiki env
```

Run Longformer IMDB classification:

```bash
make longformer imdb 1k
make longformer imdb 2k
make longformer imdb 4k
```

Run Longformer WikiText-2 MLM:

```bash
make longformer wiki 1k
make longformer wiki 2k
make longformer wiki 4k
```

The default approximate softmax profile is `HSSS-Softmax-block8`. The
Longformer results in `RESULTS_SUMMARY.md` use this default profile.

The Makefile creates and uses `model/.venv` automatically. Manual virtualenv
activation is not required.
