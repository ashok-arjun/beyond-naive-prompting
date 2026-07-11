"""
Supplementary material: CorDP (paper name: CorDP, Median or SampleWise).
Insert the following class into cik_benchmark/baselines/direct_prompt.py
after the DirectPrompt class. The codebase is built on
https://github.com/ServiceNow/context-is-key-forecasting
"""


class CorDP(DirectPrompt):
    __version__ = "0.0.7"

    def __init__(
        self,
        model,
        use_context=True,
        fail_on_invalid=True,
        n_retries=3,
        batch_size_on_retry=5,
        batch_size=None,
        constrained_decoding=True,
        token_cost: dict = None,
        temperature: float = 1.0,
        dry_run: bool = False,
        include_timestamps: bool = True,
        model_path=None,
        llm=None,
        tokenizer=None,
        custom_model=False,
        custom_model_config=None,
        resume_from_checkpoint=None,
        base_forecaster=None,
        take_median=False,
        provider=None,
        legacy=False,
        skip_system_prompt=False,
        cache_label=None,
        vllm=False,
        vllm_server_url=None,
        vllm_backend=None,
    ) -> None:
        super().__init__(model, use_context, fail_on_invalid, n_retries, batch_size_on_retry, batch_size, constrained_decoding, token_cost, temperature, dry_run, include_timestamps, model_path, llm, tokenizer, custom_model, custom_model_config, resume_from_checkpoint, provider=provider, legacy=legacy, skip_system_prompt=skip_system_prompt, vllm=vllm, vllm_server_url=vllm_server_url, vllm_backend=vllm_backend)
        self.tag = "corrected_forecast"
        self.base_forecaster = base_forecaster
        self.take_median = take_median
        self.cache_label = cache_label
        self.client = self.get_client()
        self.grand_total_cost = 0

    def make_prompt(self, task_instance, base_forecasts, max_digits=6):
        """
        Generate the prompt for the model.
        Assumes a uni-variate time series.
        """
        logger.info("Building prompt for model.")

        context = ""
        if self.use_context:
            if task_instance.all_context:
                context += f"Context: {task_instance.all_context}\n"
            else:
                if task_instance.background:
                    context += f"Background: {task_instance.background}\n"
                if task_instance.constraints:
                    context += f"Constraints: {task_instance.constraints}\n"
                if task_instance.scenario:
                    context += f"Scenario: {task_instance.scenario}\n"

        if self.include_timestamps:
            hist_time = task_instance.past_time.index.strftime("%Y-%m-%d %H:%M:%S").values
            hist_value = task_instance.past_time.values[:, -1]
            pred_time = task_instance.future_time.index.strftime("%Y-%m-%d %H:%M:%S").values
            history = "\n".join(
                f"({x}, {y:.{max_digits}g})" if y < 10**max_digits else f"({x}, {y:.0f})"
                for x, y in zip(hist_time, hist_value)
            )

            base_forecasts_with_timesteps = "\n".join(
                f"({x}, {y:.{max_digits}g})" if y < 10**max_digits else f"({x}, {y:.0f})"
                for x, y in zip(pred_time, base_forecasts[0, :, 0])
            )
            prompt = f"""
                I have a time series forecasting task for you.

                Here is some context about the task. Make sure to factor in any background knowledge,
                satisfy any constraints, and respect any scenarios.
                <context>
                {context}
                </context>

                Here is a historical time series in (timestamp, value) format:
                <history>
                {history}
                </history>

                And these are the forecasts of my statistical forecasting model in (timestamp, value) format:
                <base_forecast>
                {base_forecasts_with_timesteps}
                </base_forecast>

                My statistical forecasting model does not support taking in context as part of its input. I would like you to correct its forecasts to incorporate the context wherever necessary, and return the corrected context-aware forecast.
                Return the corrected forecast in (timestamp, value) format in between <corrected_forecast> and </corrected_forecast> tags.
                Do not include any other information (e.g., comments) in the forecast.
            """
        else:
            hist_value = task_instance.past_time.values[:, -1]
            pred_value = task_instance.future_time.values[:, -1]
            history = "\n".join(
                f"({y:.{max_digits}g})" if y < 10**max_digits else f"({y:.0f})"
                for y in hist_value
            )
            base_forecasts = "\n".join(
                f"({y:.{max_digits}g})" if y < 10**max_digits else f"({y:.0f})"
                for y in base_forecasts[0, :, 0]
            )
            prompt = f"""
                I have a time series forecasting task for you.

                Here is some context about the task. Make sure to factor in any background knowledge,
                satisfy any constraints, and respect any scenarios.
                <context>
                {context}
                </context>

                Here is a historical time series in (value) format:
                <history>
                {history}
                </history>

                And these are the forecasts of my statistical forecasting model for the next {len(pred_value)} timesteps in (value) format:
                <base_forecast>
                {base_forecasts}
                </base_forecast>

                My statistical forecasting model does not support taking in context as part of its input. I would like you to correct its forecasts to incorporate the context wherever necessary, and return the corrected context-aware forecast.
                Return the corrected forecast in (value) format in between <corrected_forecast> and </corrected_forecast> tags.
                Do not include any other information (e.g., comments) in the forecast.
            """

        return prompt

    def __call__(self, task_instance, n_samples, base_forecasts=None):
        starting_time = time.time()

        if type(base_forecasts) == type(None):
            base_forecasts, _ = self.base_forecaster(task_instance=task_instance, n_samples=n_samples)

        if self.take_median:
            base_forecasts = np.median(base_forecasts, axis=0, keepdims=True)

        prompt = self.make_prompt(task_instance, base_forecasts) if self.take_median else [self.make_prompt(task_instance, base_forecasts[i][None, :, :]) for i in range(base_forecasts.shape[0])]
        messages = [
            {
                "role": "system",
                "content": "You are a useful forecasting assistant.",
            },
            {"role": "user", "content": prompt},
        ] if self.take_median else [
            [
                {
                    "role": "system",
                    "content": "You are a useful forecasting assistant.",
                },
                {"role": "user", "content": prompt[i]},
            ] for i in range(base_forecasts.shape[0])
        ]

        if not self.take_median and (self.model.startswith("openrouter-") or self.model.startswith("gpt-") or self.vllm):
            valid_forecasts = []
            llm_outputs = []
            discarded_outputs = []
            total_tokens = {"input": 0, "output": 0}
            total_client_time = 0.0
            for message in messages:
                self.batch_size = 1
                self.batch_size_on_retry = 1
                valid_forecast, llm_output, discarded_output, tokens, client_time = self.forecast_loop(1, message, task_instance)
                valid_forecasts.extend(valid_forecast)
                llm_outputs.extend(llm_output)
                discarded_outputs.extend(discarded_output)
                total_tokens["input"] += tokens["input"]
                total_tokens["output"] += tokens["output"]
                total_client_time += client_time
        else:
            valid_forecasts, llm_outputs, discarded_outputs, total_tokens, total_client_time = self.forecast_loop(n_samples, messages, task_instance, batched_messages=self.take_median == False)

        if self.fail_on_invalid and len(valid_forecasts) < n_samples:
            raise RuntimeError(
                f"Failed to get {n_samples} valid forecasts. Got {len(valid_forecasts)} instead."
            )

        extra_info = {
            "total_input_tokens": total_tokens["input"],
            "total_output_tokens": total_tokens["output"],
            "llm_outputs": llm_outputs,
            "discarded_outputs": discarded_outputs
        }

        logger.info(f"Total tokens used: {total_tokens}")
        if self.model.startswith("openrouter-") or self.model.startswith("gpt-"):
            extra_info["input_token_cost"] = self.total_input_cost
            extra_info["output_token_cost"] = self.total_output_cost
            extra_info["total_token_cost"] = self.total_cost

            logger.info(f"Total input token cost: {self.total_input_cost}$")
            logger.info(f"Total output token cost: {self.total_output_cost}$")
            logger.info(f"Total token cost: {self.total_cost}$")

            self.grand_total_cost += self.total_cost
            self.total_input_cost = self.total_output_cost = self.total_cost = 0
            logger.info(f"Grand total token cost: {self.grand_total_cost}$")

        elif self.token_cost is not None:
            input_cost = total_tokens["input"] / 1000 * self.token_cost["input"]
            output_cost = total_tokens["output"] / 1000 * self.token_cost["output"]
            current_cost = input_cost + output_cost
            logger.info(f"Forecast cost: {current_cost}$")
            self.total_cost += current_cost

            extra_info["input_token_cost"] = self.token_cost["input"]
            extra_info["output_token_cost"] = self.token_cost["output"]
            extra_info["total_token_cost"] = current_cost

        samples = np.array(valid_forecasts)[:, :, None]

        extra_info["total_time"] = time.time() - starting_time
        extra_info["total_client_time"] = total_client_time

        gc.collect()
        del base_forecasts

        return samples, extra_info

    @property
    def cache_name(self):
        args_to_include = [
            "model",
            "use_context",
            "fail_on_invalid",
        ]
        if self.legacy:
            args_to_include.append("n_retries")
        if not self.model.startswith("gpt"):
            args_to_include.append("temperature")
        if self.custom_model:
            args_to_include.append("resume_from_checkpoint")
        if self.model_path is not None:
            args_to_include.append("model_path")
        if self.constrained_decoding is False:
            args_to_include.append("constrained_decoding")
        if self.cache_label is not None:
            args_to_include.append("cache_label")
        if self.take_median is False:
            args_to_include.append("take_median")
        if self.model.startswith("vllm-"):
            if self.vllm_backend:
                args_to_include.append("vllm_backend")
        return f"{self.__class__.__name__}_" + "_".join(
            [f"{k}={getattr(self, k)}" for k in args_to_include]
        )
