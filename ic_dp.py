"""
Supplementary material: IC-DP (paper name).
Insert the following class into cik_benchmark/baselines/direct_prompt.py
after the DirectPrompt class. The codebase is built on
https://github.com/ServiceNow/context-is-key-forecasting
"""


class IC_DP(DirectPrompt):
    __version__ = "0.0.1"

    def __init__(
        self,
        model,
        use_context=True,
        fail_on_invalid=True,
        n_retries=10,
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
        use_openrouter=False,
        provider=None,
        legacy=False,
        fewshot_task_seed=None,
        skip_system_prompt=False,
        vllm=False,
        vllm_server_url=None,
        vllm_backend=None,
        base_forecaster=None,
        take_median=False,
        cache_label=None,
    ) -> None:
        super().__init__(model, use_context, fail_on_invalid, n_retries, batch_size_on_retry, batch_size, constrained_decoding, token_cost, temperature, dry_run, include_timestamps, model_path, llm, tokenizer, custom_model, custom_model_config, resume_from_checkpoint, provider=provider, legacy=legacy, skip_system_prompt=skip_system_prompt, vllm=vllm, vllm_server_url=vllm_server_url, vllm_backend=vllm_backend)
        self.fewshot_task_seed = fewshot_task_seed
        self.base_forecaster = base_forecaster
        self.take_median = take_median
        self.tag = "corrected_forecast"
        self.cache_label = cache_label
        self.client = self.get_client()
        self.grand_total_cost = 0

    def make_prompt(self, task_instance, base_forecasts, example_task_instance, example_task_instance_base_forecasts, max_digits=6):
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

            example_hist_time = example_task_instance.past_time.index.strftime("%Y-%m-%d %H:%M:%S").values
            example_hist_value = example_task_instance.past_time.values[:, -1]
            example_pred_time = example_task_instance.future_time.index.strftime("%Y-%m-%d %H:%M:%S").values
            example_pred_value = example_task_instance.future_time.values[:, -1]
            example_task_history = "\n".join(
                f"({x}, {y:.{max_digits}g})" if y < 10**max_digits else f"({x}, {y:.0f})"
                for x, y in zip(example_hist_time, example_hist_value)
            )
            example_task_future = "\n".join(
                f"({x}, {y:.{max_digits}g})" if y < 10**max_digits else f"({x}, {y:.0f})"
                for x, y in zip(example_pred_time, example_pred_value)
            )

            example_task_instance_base_forecasts_with_timesteps = "\n".join(
                f"({x}, {y:.{max_digits}g})" if y < 10**max_digits else f"({x}, {y:.0f})"
                for x, y in zip(example_pred_time, example_task_instance_base_forecasts[0, :, 0])
            )

            prompt = f"""
                I have a context-aided time series forecasting task for you, where you will be given the history of a time series and additional context information, and prediction timesteps for which a forecast is required. You are expected to factor in any background knowledge,
                satisfy any constraints, and respect any scenarios given in the context, and output the forecast.
                in (timestamp, value) format in between <forecast> and </forecast> tags. You are to not include any other information (e.g., comments) in the forecast.

                Here is the prompt for an example task:

                Here is the context:
                <context>\nBackground: {example_task_instance.background}\nConstraints: {example_task_instance.constraints}\nScenario: {example_task_instance.scenario}\n\n</context>\n\nHere is a historical time series in (timestamp, value) format:\n<history>{example_task_history}</history>\n\nAnd these are the forecasts of my statistical forecasting model in (timestamp, value) format:\n<base_forecast>{example_task_instance_base_forecasts_with_timesteps}</base_forecast>\n\nNow please predict the value at the following timestamps: {example_pred_time}.\n

                The expected output would be:
                <corrected_forecast>{example_task_future}</corrected_forecast>

                Note how the context was incorporated in the forecast. You are expected to do the same.
                Here is the problem for which you need to return a forecast:

                Here is some context about the task.
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

                Now please predict the value at the following timestamps: {pred_time}.

                My statistical forecasting model does not support taking in context as part of its input. I would like you to correct its forecasts to incorporate the context wherever necessary, and return the corrected context-aware forecast.
                Return the corrected forecast in (value) format in between <corrected_forecast> and </corrected_forecast> tags.
                Do not include any other information (e.g., comments) in the forecast.
            """
        else:
            raise NotImplementedError("Few-shot examples are not supported for non-timestamped data yet.")

        return prompt

    def __call__(self, task_instance, n_samples, fewshot_task_instance, base_forecasts=None, fewshot_task_instance_base_forecasts=None):
        starting_time = time.time()

        if type(base_forecasts) == type(None):
            base_forecasts, _ = self.base_forecaster(task_instance=task_instance, n_samples=n_samples)
            example_task_instance_base_forecasts, _ = self.base_forecaster(task_instance=fewshot_task_instance, n_samples=n_samples)

        if self.take_median:
            base_forecasts = np.median(base_forecasts, axis=0, keepdims=True)
            example_task_instance_base_forecasts = np.median(example_task_instance_base_forecasts, axis=0, keepdims=True)

        prompt = self.make_prompt(task_instance, base_forecasts, fewshot_task_instance, example_task_instance_base_forecasts) if self.take_median else [self.make_prompt(task_instance, base_forecasts[i][None, :, :], fewshot_task_instance, example_task_instance_base_forecasts[i][None, :, :]) for i in range(base_forecasts.shape[0])]

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
        args_to_include.append("fewshot_task_seed")

        if self.vllm:
            args_to_include.append("vllm")
            args_to_include.append("vllm_server_url")
            args_to_include.append("vllm_backend")

        return f"{self.__class__.__name__}_" + "_".join(
            [f"{k}={getattr(self, k)}" for k in args_to_include]
        )
