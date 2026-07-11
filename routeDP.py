"""
routeDP: routing between a small and a large model using a router's task ordering.

RouteDP operates on existing method results. Results from any method (e.g. DP, CorDP, IC-DP)
can be passed as the small- and large-model results. The router provides an ordering of
tasks (e.g. easy to hard); at each step we assign the first k tasks to the large model
and the rest to the small model, and report aggregate metric (e.g. weighted RCRPS).

Includes random and ideal (oracle) baselines for comparison.

User provides: (1) a router, (2) small model results directory, (3) large model results
directory. The code loads results, gets the router's task ordering (using the difficulty
prompt template if the router is a callable), and runs routeDP plus baselines.
"""

import json
import os
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Same content as router_difficulty_prompt_template.txt
ROUTER_DIFFICULTY_PROMPT_TEMPLATE = """{Direct_Prompt_Prompt_of_CiK_Task}
You are given a forecasting task with full contextual information.
Please rate the task as easy or hard.
Difficulty:
"""


def _get_weighted_avg(
    per_task_metric: Dict[str, float],
    task_weights: Optional[Dict[str, float]] = None,
) -> float:
    """Weighted average of per-task metrics. Equal weights if task_weights is None."""
    if not per_task_metric:
        return 0.0
    if task_weights is None:
        task_weights = {t: 1.0 for t in per_task_metric}
    total_weight = 0.0
    weighted_sum = 0.0
    for task, value in per_task_metric.items():
        w = task_weights.get(task, 1.0)
        weighted_sum += value * w
        total_weight += w
    return float(weighted_sum / total_weight) if total_weight > 0 else 0.0


def load_router_difficulty_prompt_template(template_path: Optional[str] = None) -> str:
    """Load the router difficulty prompt template from file or return the default."""
    if template_path and os.path.isfile(template_path):
        with open(template_path, "r") as f:
            return f.read()
    return ROUTER_DIFFICULTY_PROMPT_TEMPLATE


def load_results_from_dir(results_dir: str) -> Dict[str, float]:
    """
    Load per-task average metric from a CIK-style results directory.

    Expects: results_dir / task_name / seed / "evaluation" (file containing a dict
    with "metric" key). Returns dict task_name -> mean(metric over seeds).
    """
    results = {}
    if not os.path.isdir(results_dir):
        return results
    for task_name in os.listdir(results_dir):
        task_path = os.path.join(results_dir, task_name)
        if not os.path.isdir(task_path):
            continue
        values = []
        for seed_name in os.listdir(task_path):
            seed_path = os.path.join(task_path, seed_name)
            if not os.path.isdir(seed_path):
                continue
            eval_path = os.path.join(seed_path, "evaluation")
            if not os.path.isfile(eval_path):
                continue
            try:
                with open(eval_path, "r") as f:
                    data = eval(f.read())
                if isinstance(data, dict) and "metric" in data:
                    values.append(float(data["metric"]))
            except Exception:
                continue
        if values:
            results[task_name] = float(np.mean(values))
    return results


def _router_ordering(router: Union[List[str], Dict]) -> List[str]:
    """Return ordered list of task names from router. Router can be list or dict with 'sorted_difficulty'."""
    if isinstance(router, list):
        return list(router)
    if isinstance(router, dict):
        if "sorted_difficulty" in router:
            d = router["sorted_difficulty"]
            return list(d.keys()) if isinstance(d, dict) else list(d)
        return list(router.keys())
    raise TypeError("router must be a list of task names or a dict (e.g. with 'sorted_difficulty')")


def _parse_router_response(response: str) -> float:
    """Parse router output to a difficulty score: 0 = easy, 1 = hard. Handles 'easy'/'hard' or numeric."""
    s = (response or "").strip().lower()
    if "easy" in s and "hard" not in s:
        return 0.0
    if "hard" in s:
        return 1.0
    try:
        return float(s.split()[0].replace(",", "."))
    except (ValueError, IndexError):
        return 0.5


def get_router_ordering(
    router: Union[List[str], Dict, str, Callable[[str], Any]],
    task_names: List[str],
    task_to_direct_prompt: Optional[Dict[str, str]] = None,
    prompt_template: Optional[str] = None,
) -> List[str]:
    """
    Get ordered list of task names (easy to hard) from the router.

    router: list of task names, dict with "sorted_difficulty", path to JSON file,
    or callable(prompt_str). If str and a file path, JSON is loaded.
    If callable: task_to_direct_prompt must be provided (task_name -> Direct Prompt text).
    prompt_template: format string with placeholder {Direct_Prompt_Prompt_of_CiK_Task}.
    """
    if isinstance(router, str) and os.path.isfile(router):
        with open(router, "r") as f:
            router = json.load(f)
    if not callable(router):
        return _router_ordering(router)
    if task_to_direct_prompt is None:
        raise ValueError("When router is a callable, task_to_direct_prompt must be provided.")
    template = prompt_template or ROUTER_DIFFICULTY_PROMPT_TEMPLATE
    scores = []
    for task in task_names:
        direct_prompt = task_to_direct_prompt.get(task, "")
        difficulty_prompt = template.format(Direct_Prompt_Prompt_of_CiK_Task=direct_prompt)
        response = router(difficulty_prompt)
        if isinstance(response, (int, float)):
            score = float(response)
        else:
            score = _parse_router_response(str(response))
        scores.append((task, score))
    scores.sort(key=lambda x: x[1])
    return [t for t, _ in scores]


def get_task_to_direct_prompt(seed: int = 1) -> Optional[Dict[str, str]]:
    """
    Build task_name -> Direct Prompt text using CIK benchmark and DirectPrompt.
    Returns None if cik_benchmark or DirectPrompt cannot be imported (e.g. outside repo).
    """
    try:
        from cik_benchmark import ALL_TASKS
        from cik_benchmark.baselines.direct_prompt import DirectPrompt
    except ImportError:
        return None
    dp = DirectPrompt(model="gpt-4o-mini", use_context=True, dry_run=True)
    out = {}
    for task_cls in ALL_TASKS:
        try:
            task_instance = task_cls(seed=seed)
            out[task_cls.__name__] = dp.make_prompt(task_instance)
        except Exception:
            continue
    return out if out else None


def routeDP(
    small_model: Dict[str, float],
    large_model: Dict[str, float],
    router_model: Union[List[str], Dict],
    task_weights: Optional[Dict[str, float]] = None,
) -> List[float]:
    """
    Route using the router's task ordering: assign first k tasks to large model, rest to small.

    Parameters
    ----------
    small_model : dict
        Per-task metric for the small model; keys are task names, values are metric (e.g. RCRPS).
    large_model : dict
        Per-task metric for the large model; same structure.
    router_model : list or dict
        Router ordering of tasks. Either a list of task names (e.g. easy to hard), or a dict
        containing "sorted_difficulty" (ordered task names or dict with task keys).
    task_weights : dict, optional
        Per-task weights for aggregate metric. If None, equal weights are used.

    Returns
    -------
    trajectory : list of float
        For k = 0, 1, ..., n_tasks: weighted average metric when first k tasks use large model
        and the rest use small model. Length is n_tasks + 1.
    """
    sorted_tasks = _router_ordering(router_model)
    trajectory = []
    for k in range(len(sorted_tasks) + 1):
        assignment = {
            t: large_model if i < k else small_model
            for i, t in enumerate(sorted_tasks)
        }
        combined = {t: assignment[t].get(t, 0.0) for t in sorted_tasks}
        trajectory.append(_get_weighted_avg(combined, task_weights))
    return trajectory


def random_baseline(
    small_model: Dict[str, float],
    large_model: Dict[str, float],
    n_trials: int = 100,
    seed: Optional[int] = None,
    task_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[float], List[float]]:
    """
    Random baseline: at each step randomly assign one more task to the large model.
    Averaged over n_trials.

    Parameters
    ----------
    small_model, large_model : dict
        Per-task metrics (same as routeDP).
    n_trials : int
        Number of random orderings to average.
    seed : int, optional
        Random seed.
    task_weights : dict, optional
        Per-task weights for aggregate metric.

    Returns
    -------
    mean_trajectory : list of float
        Mean aggregate metric at each step (length n_tasks + 1).
    std_trajectory : list of float
        Standard deviation at each step.
    """
    rng = np.random.default_rng(seed)
    tasks = list(small_model.keys())
    if set(tasks) != set(large_model.keys()):
        tasks = list(set(small_model) & set(large_model))
    n_tasks = len(tasks)
    if n_tasks == 0:
        return [], []

    trajectories = []
    for _ in range(n_trials):
        assignment = {t: small_model for t in tasks}
        traj = [_get_weighted_avg({t: small_model[t] for t in tasks}, task_weights)]
        unassigned = list(tasks)
        rng.shuffle(unassigned)
        for t in unassigned:
            assignment[t] = large_model
            combined = {t: assignment[t][t] for t in tasks}
            traj.append(_get_weighted_avg(combined, task_weights))
        trajectories.append(traj)
    trajectories = np.array(trajectories)
    mean_trajectory = np.mean(trajectories, axis=0).tolist()
    std_trajectory = np.std(trajectories, axis=0).tolist()
    return mean_trajectory, std_trajectory


def ideal_baseline(
    small_model: Dict[str, float],
    large_model: Dict[str, float],
    task_weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, str], List[float], List[str]]:
    """
    Ideal (oracle) baseline: greedy forward selection. At each step assign to the large
    model the task that minimizes the aggregate metric.

    Parameters
    ----------
    small_model, large_model : dict
        Per-task metrics (same as routeDP).
    task_weights : dict, optional
        Per-task weights for aggregate metric.

    Returns
    -------
    assignment : dict
        task -> "small" or "large".
    trajectory : list of float
        Aggregate metric after each greedy assignment (length n_tasks + 1, first is all-small).
    order : list of str
        Order in which tasks were assigned to the large model.
    """
    tasks = list(small_model.keys())
    if set(tasks) != set(large_model.keys()):
        tasks = list(set(small_model) & set(large_model))
    assignment = {t: "small" for t in tasks}
    trajectory = [_get_weighted_avg({t: small_model[t] for t in tasks}, task_weights)]
    order = []
    unassigned = set(tasks)

    for _ in range(len(tasks)):
        best_task = None
        best_metric = float("inf")
        for task in unassigned:
            trial = assignment.copy()
            trial[task] = "large"
            combined = {
                t: large_model[t] if trial[t] == "large" else small_model[t]
                for t in tasks
            }
            metric = _get_weighted_avg(combined, task_weights)
            if metric < best_metric:
                best_metric = metric
                best_task = task
        if best_task is None:
            break
        assignment[best_task] = "large"
        order.append(best_task)
        unassigned.remove(best_task)
        combined = {
            t: large_model[t] if assignment[t] == "large" else small_model[t]
            for t in tasks
        }
        trajectory.append(_get_weighted_avg(combined, task_weights))

    return assignment, trajectory, order


def run(
    small_results_dir: str,
    large_results_dir: str,
    router: Union[List[str], Dict, str, Callable[[str], Any]],
    task_weights: Optional[Dict[str, float]] = None,
    task_to_direct_prompt: Optional[Dict[str, str]] = None,
    prompt_template: Optional[str] = None,
    prompt_template_path: Optional[str] = None,
    n_random_trials: int = 100,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Load small and large model results, get router ordering, run routeDP and baselines.

    User provides:
      1. router: list of task names (easy to hard), path to JSON with "sorted_difficulty",
         or callable(prompt_str) that returns "easy"/"hard" or a numeric score.
      2. small_results_dir: path to small model results (CIK directory layout).
      3. large_results_dir: path to large model results.

    If router is a callable and task_to_direct_prompt is not provided, this function
    will try to build it via get_task_to_direct_prompt() when run inside the CIK repo.

    Returns dict with: routeDP_trajectory, random_mean, random_std, ideal_trajectory,
    ideal_assignment, ideal_order, sorted_tasks, small_metric, large_metric.
    """
    small_metric = load_results_from_dir(small_results_dir)
    large_metric = load_results_from_dir(large_results_dir)
    task_names = list(set(small_metric) & set(large_metric))
    if not task_names:
        return {
            "routeDP_trajectory": [],
            "random_mean": [],
            "random_std": [],
            "ideal_trajectory": [],
            "ideal_assignment": {},
            "ideal_order": [],
            "sorted_tasks": [],
            "small_metric": small_metric,
            "large_metric": large_metric,
        }

    if callable(router) and task_to_direct_prompt is None:
        task_to_direct_prompt = get_task_to_direct_prompt()
    if prompt_template is None and prompt_template_path is not None:
        prompt_template = load_router_difficulty_prompt_template(prompt_template_path)
    elif prompt_template is None:
        prompt_template = load_router_difficulty_prompt_template()

    sorted_tasks = get_router_ordering(
        router, task_names, task_to_direct_prompt=task_to_direct_prompt, prompt_template=prompt_template
    )

    routeDP_trajectory = routeDP(small_metric, large_metric, sorted_tasks, task_weights=task_weights)
    random_mean, random_std = random_baseline(
        small_metric, large_metric, n_trials=n_random_trials, seed=random_seed, task_weights=task_weights
    )
    ideal_assignment, ideal_trajectory, ideal_order = ideal_baseline(
        small_metric, large_metric, task_weights=task_weights
    )

    return {
        "routeDP_trajectory": routeDP_trajectory,
        "random_mean": random_mean,
        "random_std": random_std,
        "ideal_trajectory": ideal_trajectory,
        "ideal_assignment": ideal_assignment,
        "ideal_order": ideal_order,
        "sorted_tasks": sorted_tasks,
        "small_metric": small_metric,
        "large_metric": large_metric,
    }
