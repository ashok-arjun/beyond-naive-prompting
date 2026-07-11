# Beyond Naïve Prompting: Strategies for Improved Context-aided Forecasting with LLMs

Arjun Ashok, Andrew Robert Williams, Vincent Zhihao Zheng, Irina Rish, Nicolas Chapados, Étienne Marcotte, Valentina Zantedeschi, Alexandre Drouin

Published at TMLR (07/2026).

**Paper:** [arXiv:2508.09904](https://arxiv.org/abs/2508.09904)

Code for the methods **FxDP**, **CorDP (Median or SampleWise)**, **IC-DP**, and **routeDP**. Built on [Context-is-Key (CIK) forecasting](https://github.com/ServiceNow/context-is-key-forecasting).

**Contents:** `fxdp.py` (class FxDP), `cordp.py` (class CorDP), `ic_dp.py` (class IC_DP), `run_baselines_snippet.py` (experiment runners), `routeDP.py` (routing with random and ideal baselines), `router_difficulty_prompt_template.txt` (prompt for router difficulty rating).

# Instructions

## FxDP/CorDP/IC-DP

**1. Insert** the class code from `fxdp.py`, `cordp.py`, and `ic_dp.py` into `cik_benchmark/baselines/direct_prompt.py` after the base `DirectPrompt` class. These classes expect `DirectPrompt` to define `forecast_loop(self, n_samples, messages, task_instance, batched_messages=False)` returning `(valid_forecasts, llm_outputs, discarded_outputs, total_tokens, total_client_time)`, and `self.tag`. If the CIK `DirectPrompt` does not provide these, add them so the interface matches.

**2. Insert** the experiment functions from `run_baselines_snippet.py` into the CIK run script. Import `FxDP`, `CorDP`, `IC_DP` from `cik_benchmark.baselines.direct_prompt`, and ensure `evaluate_all_tasks`, `ChronosForecaster`, `lag_llama`, and `R_Arima` are available. Experiment names: `experiment_fxdp` (spec method `fxdp`), `experiment_cordp` (`cordp`), `experiment_ic_dp` (`ic_dp`).

**3. Run** models using the CIK repo's instructions for running baselines.

---

## routeDP

You provide only: **(1) a router**, **(2) a small model**, **(3) a large model**. As we are evaluating the method, you should also provide the results directories of (2) and (3) but in practice, the method works to route tasks to either (2) or (3). The code loads results, obtains the router's task ordering (using the difficulty prompt below if the router is a callable), and runs routeDP plus random and ideal baselines.

- **run(small_results_dir, large_results_dir, router, ...)** — Loads per-task metrics from both dirs (CIK layout: `results_dir/task_name/seed/evaluation`), gets ordered tasks from the router, runs routeDP, random_baseline, and ideal_baseline; returns trajectories and assignments.
- **router**: List of task names (easy to hard), path to a JSON with `"sorted_difficulty"`, or **callable(prompt_str)** returning "easy"/"hard" or a numeric score. For a callable, the code builds the difficulty prompt per task; if run inside the CIK repo, Direct Prompt text is obtained via **get_task_to_direct_prompt()**, else pass **task_to_direct_prompt** (task_name → Direct Prompt string).
- **router_difficulty_prompt_template.txt** — Prompt for the router. Placeholder `{Direct_Prompt_Prompt_of_CiK_Task}` is replaced with the Direct Prompt from CIK (`direct_prompt.py`). Template: `{Direct_Prompt_Prompt_of_CiK_Task}` plus "You are given a forecasting task with full contextual information. Please rate the task as easy or hard. Difficulty:"

Lower-level (when you have per-task dicts and ordering): **routeDP(...)**, **random_baseline(...)**, **ideal_baseline(...)**. **load_results_from_dir(path)** loads per-task average metric from a CIK-style directory. **get_task_to_direct_prompt(seed=1)** builds task → Direct Prompt using the CIK benchmark when run inside the repo.
