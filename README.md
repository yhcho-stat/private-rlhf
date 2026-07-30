# Privacy-Preserving RLHF via Decoupled Reward Modeling

Code for the experiments accompanying:

> **Privacy-Preserving Reinforcement Learning from Human Feedback via Decoupled Reward Modeling**  
> Young Hyun Cho and Will Wei Sun  
> Accepted in *Statistical Learning and Data Science*  
> [arXiv:2603.22563](https://arxiv.org/abs/2603.22563)

## Repository structure

```text
notebooks/
├── 01_train_private_models.ipynb
├── 02_generate_qualitative_results.ipynb
└── 03_bon_runtime.ipynb
```

The notebooks are intentionally self-contained so they can be opened and run directly in Google Colab.

| Notebook | Purpose |
|---|---|
| `01_train_private_models.ipynb` | Trains the private reward models, DP-DPO, and the split DP policy-optimization baseline; evaluates each run; and stores the experiment ledger and model artifacts. |
| `02_generate_qualitative_results.ipynb` | Loads the trained artifacts and generates the across-method qualitative comparisons and mixed-temperature Best-of-`N` outputs. |
| `03_bon_runtime.ipynb` | Measures Best-of-`N` generation and reward-scoring latency over held-out prompts. |

## Running the notebooks

The experiments were implemented for Google Colab with a CUDA GPU and Google Drive storage.

1. Open a notebook in Colab.
2. Select a GPU runtime.
3. Mount Google Drive when prompted.
4. In the first path/configuration cell, set `ROOT` to the desired Drive folder. The default is:

```python
ROOT = "/content/drive/MyDrive/Private_Finetuning"
```

5. Run the notebooks in numerical order.

The training notebook creates the following working structure under `ROOT`:

```text
Private_Finetuning/
├── master_results.csv
├── artifacts/
├── cache/
├── qualitative_generations.csv
├── qual_candidates/
├── qual_ours_mixedT.csv
├── qual_candidates_mixedT/
└── bon_latency/
```

`master_results.csv` records run configurations, privacy spending, evaluation metrics, statuses, and artifact locations. The training and analysis notebooks use this file to resume completed work and locate saved artifacts.

## Main experimental configuration

The notebooks use:

- `google/gemma-2b-it` as the base model;
- the first 40,000 examples from `Anthropic/hh-rlhf`;
- privacy levels `epsilon ∈ {0.5, 1.0, 2.0}` with `delta = 1e-5`;
- seeds `11`, `22`, and `33`;
- a maximum sequence length of 256;
- Opacus for differentially private optimization;
- last-layer LoRA for the policy baselines.

The complete settings used by each method are collected near the top of the training notebook.

## Dependencies

The notebooks install their core dependencies directly in Colab. For a separate Python environment, install:

```bash
pip install -r requirements.txt
```

## License

The code is available under the MIT License.
