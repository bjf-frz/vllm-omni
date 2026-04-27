
# Metrics

You can use these metrics in production to monitor the health and performance of the vLLM-omni system. Typical scenarios include:

- **Performance Monitoring**: Track throughput (e.g., `avg_tokens_per_s`), latency (e.g., `request_wall_time_ms` / `engine_pipeline_time_ms`), and resource utilization to verify that the system meets expected standards.

- **Debugging and Troubleshooting**: Use detailed per-request metrics to diagnose issues, such as high transfer times or unexpected token counts.

## How to Enable and View Metrics

### Start the Service with Metrics Logging

```bash
vllm serve /workspace/models/Qwen3-Omni-30B-A3B-Instruct --omni --port 8014 --log-stats
```

### Send a Request

```bash
python openai_chat_completion_client_for_multimodal_generation.py --query-type use_image
```

### What You Will See

With `--log-stats` enabled, the server will output detailed metrics logs after each request. Example output:

For multi-stage pipelines, vLLM-Omni also emits a concise per-request timing line:

```text
[OmniTiming] req=chatcmpl-a0edd05 total=32.99s preprocess=2.80s engine=30.19s stages=[0:ar=16.23s,1:diffusion=13.96s] transfers=[0->1=0.780ms]
```


#### Omni Metrics Summary

```text
============ Omni Metrics Summary ============
Successful requests:                                  1
Total E2E time (ms):                         41,356.133
Input preprocess time (ms):                      57.000
Engine pipeline time (ms):                   41,299.133
Sum check (ms):                              41,356.133

------------ Overall Time Breakdown ------------
Input preprocess time (ms):                      57.000
Stage 0 generation time (ms):                 9,910.007
Stage 0 output processor time (ms):               7.000
Stage 0 -> Stage 1 handoff time (ms):           245.895
Stage 1 generation time (ms):                30,379.198
Stage 1 output processor time (ms):              10.000
Final output time (ms):                         747.033
Component sum (ms):                          41,356.133
E2E - component sum (ms):                         0.000

------------ Stage 0 Breakdown ------------
Stage generation time (ms):                  9,910.007
Output processor time (ms):                      7.000
Stage sum check (ms):                        9,917.007

Stage id:                                            0
Stage name:                                         ar
Stage type:                                        llm
Final output type:                                text
Batch id:                                           38
Batch size:                                          1

Input tokens:                                    4,860
Output tokens:                                      67
Output token throughput (tok/s):                 6.761
Postprocess time (ms):                         256.158

------------ Stage 0 -> Stage 1 Handoff ------------
Handoff total time (ms):                       245.895
AR to diffusion time (ms):                      53.314
Other handoff processing time (ms):            192.581

------------ Final Output Breakdown ------------
Final output wrapping time (ms):                160.745
Final orchestration overhead time (ms):         586.288
Final output total time (ms):                   747.033
Final output sum check (ms):                    747.033
```

These logs include a high-level end-to-end summary, a non-overlapping stage/time breakdown, and per-stage details from `StageRequestStats`. The final output layer includes output wrapping plus the remaining orchestration overhead so the top-level component sum can be checked directly against E2E time.

You can use these logs to monitor system health, debug performance, and analyze request-level metrics as described above.


## Metrics Scope: Offline vs Online Inference

For **offline inference** (batch mode), the summary includes both system-level metrics (aggregated across all requests) and per-request metrics. In this case, `num_of_requests` can be greater than 1, reflecting multiple completed requests in a batch.

For **online inference** (serving mode), the summary is always per-request. `num_of_requests` is always 1, and only request-level metrics are reported for each completion.

When `num_of_requests` is 1, average fields are omitted from the returned overall summary because they are identical to the total/request-level values. They remain meaningful for offline batches with multiple completed requests.

---

## Parameter Details

### Summary Metrics

| Field                     | Meaning                                                                                       |
|---------------------------|----------------------------------------------------------------------------------------------|
| `num_of_requests`         | Number of completed requests.                                                                |
| `request_wall_time_ms`        | Wall-clock time span from request preparation start to final completion, in ms.          |
| `input_preprocess_time_ms` | Time spent preparing and submitting requests before the engine pipeline starts.             |
| `engine_pipeline_time_ms` | Time from engine request submission to final completion.                                     |
| `stage_gen_total_time_ms` | Sum of all stage `stage_gen_time_ms` values.                                                 |
| `output_processor_total_time_ms` | Sum of time spent in per-stage output processors after raw engine outputs are returned. |
| `stage_handoff_total_time_ms` | Sum of inter-stage handoff time, measured after an upstream stage finishes and before the downstream stage is submitted. |
| `ar2diffusion_total_time_ms` | Subset of `stage_handoff_total_time_ms` spent converting AR outputs into diffusion inputs. |
| `final_output_total_time_ms` | Measured time spent wrapping final stage outputs into `OmniRequestOutput` objects before yielding them. |
| `breakdown_delta_time_ms` | Difference between E2E wall time and the printed component sum; useful for detecting missing timers or overlap. |
| `stage_{i}_to_{j}_handoff_time_ms` | Handoff time for a specific stage edge, e.g. `stage_0_to_1_handoff_time_ms`.       |
| `stage_{i}_to_{j}_ar2diffusion_time_ms` | AR-to-diffusion conversion time included in that edge's handoff time.           |
| `total_tokens`            | Total tokens counted across all completed requests (stage0 input + all stage outputs).       |
| `avg_request_wall_time_ms` | Average wall time per request: `request_wall_time_ms / num_of_requests`.                  |
| `avg_input_preprocess_time_ms` | Average pre-submit request preparation time per completed request.                    |
| `avg_engine_pipeline_time_ms` | Average engine pipeline time per completed request.                                      |
| `avg_stage_gen_total_time_ms` | Average summed stage generation time per completed request.                              |
| `avg_output_processor_time_ms` | Average output processor time per completed request.                                   |
| `avg_stage_handoff_total_time_ms` | Average summed inter-stage handoff time per completed request.                    |
| `avg_ar2diffusion_time_ms` | Average AR-to-diffusion conversion time per completed request.                             |
| `avg_final_output_time_ms` | Average final output wrapping time per completed request.                                  |
| `avg_breakdown_delta_time_ms` | Average difference between E2E wall time and the printed component sum per completed request. |
| `avg_tokens_per_s`        | Average token throughput over wall time: `total_tokens * 1000 / request_wall_time_ms`.      |
| `stage_{i}_wall_time_ms`  | Wall-clock time span for stage i, in ms. Each stage's wall time is reported as a separate field, e.g., `stage_0_wall_time_ms`, `stage_1_wall_time_ms`, etc. |

### Timing Breakdown

The printed summary includes sum checks that show the containment relationship:

`request_wall_time_ms = input_preprocess_time_ms + engine_pipeline_time_ms`

`request_wall_time_ms ~= input_preprocess_time_ms + sum(stage_gen_time_ms) + sum(output_processor_time_ms) + sum(stage_handoff_time_ms) + final_output_total_time_ms`

Any remaining difference is reported as `breakdown_delta_time_ms`.

`ar2diffusion_total_time_ms` is included in `stage_handoff_total_time_ms`. For a concrete AR-to-diffusion edge, `stage_0_to_1_ar2diffusion_time_ms` is included in `stage_0_to_1_handoff_time_ms`.

For offline batches, average component fields such as `avg_stage_gen_total_time_ms`,
`avg_output_processor_time_ms`, `avg_stage_handoff_total_time_ms`, and
`avg_ar2diffusion_time_ms` are computed by dividing the batch aggregate by
`num_of_requests`. They are not individually timed per request and then averaged.
Per-request timers remain available in the E2E table where they are measured directly.

---

### E2E Table (per request)

| Field                     | Meaning                                                               |
|---------------------------|-----------------------------------------------------------------------|
| `request_wall_time_ms`    | End-to-end latency in ms, including input preprocessing and engine pipeline time.            |
| `input_preprocess_time_ms` | Time spent preparing and submitting the request before `engine_pipeline_time_ms` starts.    |
| `engine_pipeline_time_ms` | Time from engine request submission to final completion.                                     |
| `total_tokens`            | Total tokens for the request (stage0 input + all stage outputs).      |
| `transfers_total_time_ms` | Sum of transfer edge `total_time_ms` for this request.                |
| `transfers_total_kbytes`  | Sum of transfer kbytes for this request.                              |
| `final_output_time_ms`    | Time spent wrapping the final engine output into an `OmniRequestOutput` before yielding it. |


---

### Stage Table (per stage event / request)

| Field                     | Meaning                                                                                         |
|---------------------------|-------------------------------------------------------------------------------------------------|
| `batch_id`                | Batch index.                                                                                    |
| `batch_size`              | Batch size.                                                                                     |
| `num_tokens_in`           | Input tokens to the stage.                                                                      |
| `num_tokens_out`          | Output tokens from the stage.                                                                   |
| `stage_gen_time_ms`       | Stage compute time in ms, excluding output processor time and postprocessing time.             |
| `output_processor_time_ms` | Time spent in the stage output processor after raw engine outputs are returned.              |
| `image_num`               | Number of images generated (for diffusion/image stages).                                        |
| `resolution`              | Image resolution (for diffusion/image stages).                                                                  |
| `postprocess_time_ms` | Diffusion/image: post-processing time in ms.                                                    |

---

### Transfer Table (per edge / request)

| Field                | Meaning                                                                   |
|----------------------|---------------------------------------------------------------------------|
| `size_kbytes`        | Total kbytes transferred.                                                 |
| `tx_time_ms`         | Sender transfer time in ms.                                               |
| `rx_decode_time_ms`  | Receiver decode time in ms.                                               |
| `in_flight_time_ms`  | In-flight time in ms.                                                     |


### Expectation of the Numbers (Verification)

**Formulas:**

- `total_tokens = Stage0's num_tokens_in + sum(all stages' num_tokens_out)`

- `transfers_total_time_ms = sum(tx_time_ms + rx_decode_time_ms + in_flight_time_ms)` for every edge

**Using the example above:**

**total_tokens**

- Stage0's `num_tokens_in`: **4,860**
- Stage0's `num_tokens_out`: **67**
- Stage1's `num_tokens_out`: **275**
- Stage2's `num_tokens_out`: **0**

so `total_tokens = 4,860 + 67 + 275 + 0 = 5,202`, which matches the table value `total_tokens`.

**transfers_total_time_ms**

For each edge:

- 0->1: tx_time_ms (**78.701**) + rx_decode_time_ms (**111.865**) + in_flight_time_ms (**2.015**) = **192.581**

- 1->2: tx_time_ms (**18.790**) + rx_decode_time_ms (**31.706**) + in_flight_time_ms (**2.819**) = **53.315**

192.581 + 53.315 = **245.896** = transfers_total_time_ms, which matches the calculation (difference is due to rounding)
