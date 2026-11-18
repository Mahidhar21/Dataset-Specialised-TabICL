# Dataset-Specialised TabICL

Course project: dataset representations, cluster-based expert specialisation, and retrieval-based prediction, together with a separate TALENT benchmark of pretrained TabICL v1.

## Repository layout

- `dataset_representation_and_clustering/`: original statistical and TabICL representations, stability experiments, clustering scripts and saved exploratory outputs.
- `expert_training_and_routing/`: final 512D/K=4 pipeline, manifest, specialist training, evaluation and routing ablations.
- `talent_benchmarking/`: independent v1 Colab notebook, with its own environment.
- `results/`: saved v2 histories/evaluations and supplied TALENT workbook and report.
- `report/`: consolidated IEEE LaTeX manuscript and figures. Compile `main.tex` in Overleaf.

Original Stage 1–2 work: https://github.com/Mahidhar21/Dataset-Specialised-TabICL . Stage 3–4 implementation and benchmarking notebook source: https://github.com/sri-nikesh-31/Dataset-Specialised-TabICL . Retain original contributor attribution and licenses.

## Reproduce the v2 specialist experiment

Run from the repository root. Use Python 3.10 and a CUDA-enabled PyTorch installation appropriate to your GPU. `requirements.txt` records the original environment; the CUDA build may need its matching PyTorch package index. The trainer relies on internal TabICL APIs: use the upstream commit in `tabicl_upstream_commit.txt`, not the v1 notebook commit.

```bash
python -m venv .venv
source .venv/bin/activate
# Install the matching CUDA PyTorch wheel first if needed.
python -m pip install -r requirements.txt
python -m pip install matplotlib
# Pin upstream TabICL (fine-tuning extras).
python -m pip install "tabicl[finetune] @ git+https://github.com/soda-inria/tabicl.git@$(cat tabicl_upstream_commit.txt)"
export DATA=/absolute/path/to/dataset-250115/data
```

Original datasets are not distributed here. Use the original numerical dataset arrays (`N_train.npy`, `y_train.npy`) as named in the committed manifest. To regenerate the exploratory Stage 1–2 artifacts, use a separate checkout if preserving saved outputs:

```bash
python dataset_representation_and_clustering/strategy_3_1.py
python dataset_representation_and_clustering/strategy_3_2.py
python dataset_representation_and_clustering/test_embedding_stability.py
python dataset_representation_and_clustering/compare_representations.py
python dataset_representation_and_clustering/prepare_routing_experiment.py
python dataset_representation_and_clustering/prepare_routing_k4.py
```

For the final specialist run, use the frozen manifest (53 meta-train, 14 meta-test) and reproduce the 512D router. Do not ignore cluster-assignment mismatches.

```bash
python -m expert_training_and_routing.extract_embeddings --data-dir "$DATA" --device cpu
python -m expert_training_and_routing.build_router
python -m expert_training_and_routing.train_experts --data-dir "$DATA" --expert 0 --epochs 1 --steps-per-epoch 5 --rows 64 --features 32 --output outputs/stage32_smoke/experts
for i in 0 1 2 3; do
  python -m expert_training_and_routing.train_experts --data-dir "$DATA" --expert "$i" --epochs 3 --steps-per-epoch 30 --rows 64 --features 32 || break
done
python -m expert_training_and_routing.evaluate --data-dir "$DATA" --max-datasets 0 --top-k 3
for mode in nearest random fixed; do
  python -m expert_training_and_routing.evaluate_routing_ablation --data-dir "$DATA" --routing "$mode" --seed 42 --top-k 3 --fixed-experts 0,1,2 --output "outputs/stage32/ablation_$mode" || break
done
```

Generated checkpoints, embeddings and router are omitted from Git. Preserve the verified original baseline archive separately. The v2 evaluation samples context/query rows within the original training partition, not the official test partition. The original preprocessing imputes from the complete source training array before sampling. The router uses standardised embeddings and Euclidean centroid distance, not cosine similarity.

## Results

| v2 method (14 held-out datasets) | Accuracy | Balanced accuracy | Macro F1 | Log loss |
|---|---:|---:|---:|---:|
| Pretrained | 86.16% | 82.39% | 82.02% | 0.28946 |
| Nearest Top-1 | 85.71% | 80.79% | 80.76% | 0.29900 |
| Nearest Top-3 | 86.33% | 81.76% | 81.44% | 0.29054 |
| Random Top-3 | 86.22% | 81.12% | 80.92% | 0.29056 |
| Fixed Top-3 | 86.38% | 81.88% | 81.64% | 0.29105 |

The current results demonstrate feasibility, not a reliable routing advantage. Expert 2 retains its pretrained checkpoint after validation selection.

## Independent TALENT v1 benchmark

See [notebook instructions](talent_benchmarking/README.md). The report describes 188/200 completed tasks, mean accuracy 83.48%, mean macro OvR AUC 0.8936, mean log loss 0.3796 and median fit-plus-predict time 12.57 s. The workbook has 187 scored tasks (earlier snapshot). The benchmark and specialist results use different protocols and must not be directly compared as a model improvement.
