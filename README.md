# sgexams-data-analysis

This repository contains data collected from the r/SGExams subreddit from July 2024 to March 2025. 

Also, it contains code to train ML models to classify if a thread should be marked as "popular". We define a "popular" thread as one that received upvotes in the top 25% of all threads in the dataset. For this dataset, the thread must have received at least 41 upvotes.

## Data

The data was collected using the [Python Reddit API Wrapper](https://praw.readthedocs.io/en/stable/) by running a daily cron job. Also the dataset has been filtered to remove:
1. All data which has been marked as removed by the official [Python Reddit API Wrapper](https://praw.readthedocs.io/en/stable/)
2. All data which contains media (`is_self=False`)

There are 4 datasets in the `data` folder:
1. `data/threads_jul24_mar25.csv` = Full dataset containing 9,703 rows
2. `data/threads_jul24_mar25_train.csv` = Training dataset containing 7,762 rows (80%)
3. `data/threads_jul24_mar25_val.csv` = Validation dataset containing 970 rows (10%)
4. `data/threads_jul24_mar25_test.csv` = Test dataset containing 971 rows (10%)


## ML Classifier

In this section, we outline how to run the experiments to train your own ML classifier and to replicate the results found in `data/model_comparison_results.csv`. `data/model_comparison_results.csv` contains the results of all experiments conducted.

### 1. Install dependencies

This project uses Python 3.12. It is recommended to create a virtual environment to install the dependencies.

```bash
pip install -r requirements.txt
```

### 2. Run the experiments

To run all experiments and replicate the results found in `data/model_comparison_results.csv`, run the following command:

```bash
python main.py --all --test
```

To run the experiment with the best subset of features, run the following command:

```bash
python main.py
```

To run the experiment with **all** features, run the following command:

```bash
python main.py --all
```

A summary of all command line arguments is shown below:

| Flag               | Type               | Default   | Variable                                 | Description                                                                                                           |
|--------------------|--------------------|-----------|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `--mode {0,1}`     | choice (`0`/`1`)   | `0`       | `TRAIN_BEST_MODEL`, `TRAIN_ALL_FEATURES` | Selects which model to train:<br>- `0`: train best model (default)<br>- `1`: train model with **all** features        |
| `--all`            | boolean flag       | `false`   | `RUN_ALL_EXPERIMENTS`                    | If set, **all** experiments will be run (overrides `--mode`).<br>_Warning: may cause a segfault if memory is low._    |
| `--test`           | boolean flag       | `false`   | `GET_TEST_PERFORMANCE`                   | If set, evaluate on the **test** dataset; otherwise only validation is performed.                                     |

