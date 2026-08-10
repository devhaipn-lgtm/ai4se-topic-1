# Topic 1: Issue Report Classification

## Purpose

The project predicts whether a GitHub issue is a `bug`, `feature`, or
`question`. The NLBSE'24 dataset contains 3,000 issues from five repositories.
We train one chosen text model for each repository using the issue title and
description.

Five text configurations and one dummy baseline were compared with five-fold
cross-validation on the training set. The chosen configuration is word-and-
character TF-IDF with LinearSVC. The official test set is used only once, for
the final evaluation.

The four numbered notebooks are the main project workflow. Run them in order.
They call tested functions from `src/issue_classifier` and display the saved
inspection, model-selection, evaluation, and analysis results.

## Results

| Result | Cross-repository macro-F1 |
| --- | ---: |
| Our chosen model | 0.754385 |
| NLBSE'24 detailed SetFit result | 0.823973 |
| NLBSE'24 headline SetFit result | 0.827000 |
| NLBSE'24 RoBERTa result | 0.792000 |
| NLBSE'24 fastText result | 0.718000 |

The chosen model reaches 0.754385 on the official test set. React is the
strongest repository at 0.807182. Bitcoin is the weakest at 0.700619.
`question` is the hardest label, with F1 of 0.710436.

## Folder structure

Our work is at the top level. Files copied from the official NLBSE'24 repository
are isolated in `official_nlbse24/`.

```text
AI4SE_FINAL_PROJECT/
|-- README.md                      Launch and result guide
|-- requirements.txt              Python dependencies
|-- src/issue_classifier/          Our implementation
|-- tests/                         Automated tests
|-- notebooks/                     Four executed notebooks, run in order
|-- results/chosen-model-seed42/   Chosen-model metrics and predictions
|-- report/FINAL_REPORT.pdf        Complete report
`-- official_nlbse24/              Original data and published references
```

## Recommended workflow: Colab notebooks

1. Upload the complete `AI4SE_FINAL_PROJECT` folder to Google Drive.
2. Open the notebooks from `notebooks/` in numerical order.
3. Select **Runtime > Run all** for each notebook.

The notebooks locate the project folder automatically. If Colab has not loaded
the dependencies yet, run this once in a code cell:

```python
%pip install -r /content/drive/MyDrive/AI4SE_FINAL_PROJECT/requirements.txt
```

The notebooks already contain executed outputs. Normal execution reads and
displays the saved results. Training runs only when an optional rerun switch is
changed manually.

## Recommended workflow: local notebooks

From this folder, use Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter lab
```

Environment activation is not required. Calling
`.venv\Scripts\python.exe` directly guarantees that every command uses the
project's Python 3.11 environment.

Open `notebooks/01_eda.ipynb`, then continue through notebooks 02, 03, and 04.
Select **Run All Cells** in each notebook.

## Optional command-line execution

The notebooks are the main presentation path. These commands provide automated
checks and full experiment reruns from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m issue_classifier validate-data --backend file
.\.venv\Scripts\python.exe -m issue_classifier validate-data --backend memory
.\.venv\Scripts\python.exe -m issue_classifier cross-validate --model all
.\.venv\Scripts\python.exe -m issue_classifier train --model selected
.\.venv\Scripts\python.exe -m issue_classifier evaluate --model selected
```

## Main files

- Model selection: `results/cross_validation.json`
- Chosen-model metrics: `results/chosen-model-seed42/metrics.json`
- Chosen-model predictions: `results/chosen-model-seed42/predictions.csv`
- Final paper: `report/FINAL_REPORT.pdf`
- LaTeX source: `report/main.tex`

## Notebook sequence

All four notebooks contain executed outputs and clearly labelled steps.

1. `notebooks/01_eda.ipynb`: preprocessing, balance, missing values, duplicates,
   and exact train/test overlaps.
2. `notebooks/02_chosen_model.ipynb`: the six-candidate, five-repository,
   five-fold comparison and the selected configuration.
3. `notebooks/03_final_evaluation.ipynb`: how the selected configuration is
   fitted separately for five repositories and applied to the test set.
4. `notebooks/04_results.ipynb`: final metrics, confusion matrix, repository
   comparison, label comparison, and error analysis.

## Experimental rules

- Model selection uses training data only.
- The official test set is used only for final evaluation.
- Repository name and creation time are not model features.
- The primary score includes all 1,500 test issues.
- A secondary analysis removes only the three exact train/test overlaps.
