"""
Supplementary material: FxDP (paper name).
Insert the following class into cik_benchmark/baselines/direct_prompt.py
after the DirectPrompt class. The codebase is built on
https://github.com/ServiceNow/context-is-key-forecasting
"""


class FxDP(DirectPrompt):
    __version__ = "0.1.0"

    def make_prompt(self, task_instance, max_digits=6):
        """
        Generate the prompt for the model.
        Assumes a uni-variate time series.
        """
        logger.info("Building prompt for model.")

        # Extract context
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

                You are tasked with predicting the value at the following timestamps: {pred_time}.

                First, within <reason> and </reason> tags, walk-through step-by-step how you would incorporate each piece of the context to improve your forecast. If you think any of the context is irrelevant, please indicate. At the end, state the effect of the context on the forecast by stating `Therefore, the effect of the context on the forecast would be that` continued by the effect of the context on the forecast.

                Next, return your forecast in (timestamp, value) format in between <forecast> and </forecast> tags.
                Do not include any other information (e.g., comments) in the forecast.

                Example:
                <history>
                (t1, v1)
                (t2, v2)
                (t3, v3)
                </history>
                <forecast>
                (t4, v4)
                (t5, v5)
                </forecast>

            """
        else:
            hist_value = task_instance.past_time.values[:, -1]
            pred_value = task_instance.future_time.values[:, -1]
            history = "\n".join(
                f"({y:.{max_digits}g})" if y < 10**max_digits else f"({y:.0f})"
                for y in hist_value
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

                You are tasked with predicting the value for the next {len(pred_value)} timesteps.

                First, within <reason> and </reason> tags, walk-through step-by-step how you would incorporate each piece of the context to improve your forecast. If you think any of the context is irrelevant, please indicate. At the end, state the effect of the context on the forecast by stating `Therefore, the effect of the context on the forecast would be that` continued by the effect of the context on the forecast.

                Next, return your forecast in (timestamp, value) format in between <forecast> and </forecast> tags.
                Do not include any other information (e.g., comments) in the forecast.

                Example:
                <history>
                (v1)
                (v2)
                (v3)
                </history>
                <forecast>
                (v4)
                (v5)
                </forecast>

            """

        return prompt

    def forecast_loop(self, n_samples, messages, task_instance):
        default_batch_size = n_samples if not self.batch_size else self.batch_size
        if self.batch_size:
            assert (
                self.batch_size * self.n_retries >= n_samples
            ), f"Not enough iterations to cover {n_samples} samples"
        assert (
            self.batch_size_on_retry <= default_batch_size
        ), f"Batch size on retry should be equal to or less than {default_batch_size}"

        max_batch_size = task_instance.max_directprompt_batch_size
        if max_batch_size is not None:
            batch_size = min(default_batch_size, max_batch_size)
            n_retries = self.n_retries + default_batch_size // batch_size
        else:
            batch_size = default_batch_size
            n_retries = self.n_retries

        llm_outputs = []
        discarded_outputs = []
        valid_forecasts = []
        valid_reasons = []
        total_tokens = {"input": 0, "output": 0}
        total_client_time = 0.0

        while len(valid_forecasts) < n_samples and n_retries > 0:
            logger.info(f"Requesting forecast of {batch_size} samples from the model.")
            client_start_time = time.time()

            if self.include_timestamps:
                if "future_timestamps" in inspect.signature(self.client).parameters:
                    chat_completion = self.client(
                        model=self.model,
                        n=batch_size,
                        messages=messages,
                        future_timestamps=task_instance.future_time.index.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ).values,
                        include_reason=True
                    )
                else:
                    chat_completion = self.client(
                        model=self.model, n=batch_size, messages=messages
                    )
            else:
                if "pred_length" in inspect.signature(self.client).parameters:
                    chat_completion = self.client(
                        model=self.model,
                        n=batch_size,
                        messages=messages,
                        pred_length=len(task_instance.future_time)
                    )
                else:
                    chat_completion = self.client(
                        model=self.model, n=batch_size, messages=messages
                    )

            total_client_time += time.time() - client_start_time
            current_input_tokens = chat_completion.usage.prompt_tokens
            current_output_tokens = chat_completion.usage.completion_tokens
            current_cost = chat_completion.usage.cost

            total_tokens["input"] += chat_completion.usage.prompt_tokens
            total_tokens["output"] += chat_completion.usage.completion_tokens

            logger.info("Parsing forecasts from completion.")
            for choice in chat_completion.choices:
                try:
                    forecast = extract_html_tags(choice.message.content, [self.tag])[
                        self.tag
                    ][0]
                    forecast = forecast.replace("(", "").replace(")", "")
                    forecast = forecast.split("\n")

                    reason = extract_html_tags(choice.message.content, ["reason"])[
                        "reason"
                    ][0]

                    if self.include_timestamps:
                        forecast = {
                            x.split(",")[0]
                            .replace("'", "")
                            .replace('"', ""): float(x.split(",")[1])
                            for x in forecast
                        }
                        forecast = [
                            forecast[t]
                            for t in task_instance.future_time.index.strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        ]
                    else:
                        forecast = [float(x) for x in forecast]
                        if len(forecast) != len(task_instance.future_time):
                            raise Exception()

                    valid_reasons.append(reason)
                    valid_forecasts.append(forecast)
                    llm_outputs.append(choice.message.content)

                except Exception as e:
                    logger.info("Sample rejected due to invalid format.", choice.message.content)
                    logger.debug(f"Rejection details: {e}")
                    logger.debug(f"Choice: {choice.message.content}")
                    discarded_outputs.append(choice.message.content)

            n_retries -= 1
            if max_batch_size is not None:
                remaining_samples = n_samples - len(valid_forecasts)
                batch_size = max(remaining_samples, self.batch_size_on_retry)
                batch_size = min(batch_size, max_batch_size)
            else:
                batch_size = self.batch_size_on_retry

            valid_reasons = valid_reasons[:n_samples]
            valid_forecasts = valid_forecasts[:n_samples]
            logger.info(f"Got {len(valid_forecasts)}/{n_samples} valid forecasts.")
            if len(valid_forecasts) < n_samples:
                logger.info(f"Remaining retries: {n_retries}.")

            if self.model and self.model.startswith("openrouter-") or self.model.startswith("gpt-"):
                model_name = self.model
                if self.model.startswith("openrouter-"):
                    if current_cost:
                        input_cost = output_cost = 0
                        logger.info(f"Current forecast cost directly from OpenRouter: {current_cost}$")
                    else:
                        logger.info(f"Cost not recorded. Computing.")
                        provider = chat_completion.provider
                        model_name = self.model + "-" + provider
                        if model_name in OPENROUTER_COSTS:
                            input_cost = (
                                current_input_tokens
                                / 1000
                                * OPENROUTER_COSTS[model_name]["input"]
                            )
                            output_cost = (
                                current_output_tokens
                                / 1000
                                * OPENROUTER_COSTS[model_name]["input"]
                            )
                            current_cost = input_cost + output_cost
                            logger.info(f"Current forecast cost - computed: {current_cost}$")
                elif model_name in GPT_COSTS:
                    input_cost = (
                        current_input_tokens
                        / 1000
                        * GPT_COSTS[model_name]["input"]
                    )
                    output_cost = (
                        current_output_tokens
                        / 1000
                        * GPT_COSTS[model_name]["output"]
                    )
                    current_cost = input_cost + output_cost
                    logger.info(f"Current forecast cost - computed: {current_cost}$")
                else:
                    input_cost = output_cost = current_cost = 0
                    logger.info(f"Cost not recorded")

                self.total_input_cost += input_cost
                self.total_output_cost += output_cost
                self.total_cost += current_cost

        return valid_forecasts, valid_reasons, llm_outputs, discarded_outputs, total_tokens, total_client_time

    def __call__(self, task_instance, n_samples):
        starting_time = time.time()

        prompt = self.make_prompt(task_instance)
        messages = [
            {
                "role": "system",
                "content": "You are a useful forecasting assistant.",
            },
            {"role": "user", "content": prompt},
        ] if not self.skip_system_prompt else [
            {"role": "user", "content": "You are a useful forecasting assistant." + prompt}
        ]

        valid_forecasts, valid_reasons, llm_outputs, discarded_outputs, total_tokens, total_client_time = self.forecast_loop(n_samples, messages, task_instance)

        if self.fail_on_invalid and len(valid_forecasts) < n_samples:
            raise RuntimeError(
                f"Failed to get {n_samples} valid forecasts. Got {len(valid_forecasts)} instead."
            )

        extra_info = {
            "total_input_tokens": total_tokens["input"],
            "total_output_tokens": total_tokens["output"],
            "llm_outputs": llm_outputs,
            "discarded_outputs": discarded_outputs,
            "valid_reasons": valid_reasons
        }

        logger.info(f"Total tokens used: {total_tokens}")
        if self.model.startswith("openrouter-") or self.model.startswith("gpt-"):
            extra_info["input_token_cost"] = self.total_input_cost
            extra_info["output_token_cost"] = self.total_output_cost
            extra_info["total_token_cost"] = self.total_cost

            logger.info(f"Total input token cost: {self.total_input_cost}$")
            logger.info(f"Total output token cost: {self.total_output_cost}$")
            logger.info(f"Total token cost: {self.total_cost}$")
            self.total_input_cost = self.total_output_cost = self.total_cost = 0

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

        return samples, extra_info
