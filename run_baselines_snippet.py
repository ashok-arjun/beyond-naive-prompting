"""
Supplementary material: experiment runners for FxDP, CorDP, IC-DP.
Insert the following into the run script (e.g. run_baselines.py) in the
Context-is-Key (CIK) forecasting repo. Ensure DirectPrompt, FxDP, CorDP, IC_DP
are imported from cik_benchmark.baselines.direct_prompt, and that
evaluate_all_tasks, ChronosForecaster, lag_llama, R_Arima are available.
See README for full instructions.
Reference: https://github.com/ServiceNow/context-is-key-forecasting
"""


def experiment_fxdp(
    llm,
    use_context,
    n_samples,
    output_folder,
    max_parallel=1,
    skip_cache_miss=False,
    batch_size=None,
    batch_size_on_retry=5,
    n_retries=35,
    temperature=1.0,
    custom_model=False,
    custom_model_config=None,
    resume_from_checkpoint=None,
    constrained_decoding=True,
    n_seeds=5,
    task=None,
    seed=None,
    provider=None,
    roi_tasks_only=False,
    max_tokens=10000,
):
    """FxDP."""
    forecaster = FxDP(
        model=llm,
        use_context=use_context,
        token_cost={"input": 0.0, "output": 0.0},
        batch_size=batch_size,
        batch_size_on_retry=batch_size_on_retry,
        n_retries=n_retries,
        temperature=temperature,
        dry_run=skip_cache_miss,
        custom_model=custom_model,
        custom_model_config=custom_model_config,
        resume_from_checkpoint=resume_from_checkpoint,
        constrained_decoding=constrained_decoding,
        provider=provider,
        max_tokens=max_tokens,
    )
    results = evaluate_all_tasks(
        forecaster,
        n_samples=n_samples,
        output_folder=f"{output_folder}/{forecaster.cache_name}",
        max_parallel=max_parallel,
        skip_cache_miss=skip_cache_miss,
        seeds=n_seeds,
        use_cache=False if custom_model else True,
        task=task,
        seed=seed,
        roi_tasks_only=roi_tasks_only,
    )
    del forecaster
    return results, {}


def experiment_cordp(
    llm,
    use_context,
    n_samples,
    output_folder,
    max_parallel=1,
    skip_cache_miss=False,
    batch_size=None,
    batch_size_on_retry=5,
    n_retries=3,
    temperature=1.0,
    custom_model=False,
    custom_model_config=None,
    resume_from_checkpoint=None,
    constrained_decoding=True,
    n_seeds=5,
    take_median=True,
    task=None,
    seed=None,
    provider=None,
    legacy=False,
):
    """CorDP (Median or SampleWise)."""
    base_forecaster = ChronosForecaster(model_size="large")
    forecaster = CorDP(
        model=llm,
        use_context=use_context,
        token_cost={"input": 0.0, "output": 0.0},
        batch_size=batch_size,
        batch_size_on_retry=batch_size_on_retry,
        n_retries=n_retries,
        temperature=temperature,
        dry_run=skip_cache_miss,
        custom_model=custom_model,
        custom_model_config=custom_model_config,
        resume_from_checkpoint=resume_from_checkpoint,
        constrained_decoding=constrained_decoding,
        base_forecaster=base_forecaster,
        take_median=take_median,
        provider=provider,
        legacy=legacy,
    )
    results = evaluate_all_tasks(
        forecaster,
        n_samples=n_samples,
        output_folder=f"{output_folder}/{forecaster.cache_name}",
        max_parallel=max_parallel,
        skip_cache_miss=skip_cache_miss,
        task=task,
        seed=seed,
        seeds=n_seeds,
    )
    total_cost = forecaster.total_cost
    del forecaster
    return results, {"total_cost": total_cost}


def experiment_ic_dp(
    llm,
    use_context,
    n_samples,
    output_folder,
    max_parallel=1,
    skip_cache_miss=False,
    batch_size=None,
    batch_size_on_retry=5,
    n_retries=3,
    temperature=1.0,
    custom_model=False,
    custom_model_config=None,
    resume_from_checkpoint=None,
    constrained_decoding=True,
    n_seeds=5,
    task=None,
    seed=None,
    provider=None,
    legacy=False,
    fewshot_task_seed=500,
    vllm=False,
    vllm_server_url=None,
    vllm_backend=None,
    base_model="lag_llama",
    cache_label=None,
    take_median=True,
    roi_tasks_only=False,
):
    """IC-DP."""
    if base_model == "lag_llama":
        base_forecaster = lag_llama
    elif base_model == "chronos_forecaster_large":
        base_forecaster = ChronosForecaster(model_size="large")
    elif base_model == "r_arima":
        base_forecaster = R_Arima()
    else:
        raise ValueError(f"Invalid base model: {base_model}.")

    forecaster = IC_DP(
        model=llm,
        use_context=use_context,
        token_cost={"input": 0.0, "output": 0.0},
        batch_size=batch_size,
        batch_size_on_retry=batch_size_on_retry,
        n_retries=n_retries,
        temperature=temperature,
        dry_run=skip_cache_miss,
        custom_model=custom_model,
        custom_model_config=custom_model_config,
        resume_from_checkpoint=resume_from_checkpoint,
        constrained_decoding=constrained_decoding,
        provider=provider,
        legacy=legacy,
        fewshot_task_seed=fewshot_task_seed,
        vllm=vllm,
        vllm_server_url=vllm_server_url,
        vllm_backend=vllm_backend,
        base_forecaster=base_forecaster,
        cache_label=cache_label,
        take_median=take_median,
    )
    results = evaluate_all_tasks(
        forecaster,
        n_samples=n_samples,
        output_folder=f"{output_folder}/{forecaster.cache_name}",
        max_parallel=max_parallel,
        skip_cache_miss=skip_cache_miss,
        seeds=n_seeds,
        use_cache=False if custom_model else True,
        task=task,
        seed=seed,
        fewshot_task_seed=fewshot_task_seed,
        roi_tasks_only=roi_tasks_only,
    )
    total_cost = forecaster.total_cost
    del forecaster
    return results, {"total_cost": total_cost}
