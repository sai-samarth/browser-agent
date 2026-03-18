# Browser-Agent Baseline Report

## Corpus snapshot

- rollout runs discovered: 45
- runs with episode summaries: 31
- runs with step traces: 37
- total episodes: 1027
- total steps: 5505
- total successful episodes: 436
- overall success rate across all logged episodes: 42.45%

## Run classification summary

- SFT candidates: 1
- partial runs: 2
- broken runs: 11

## Best accuracy baseline

- run: `miniwob_train_zai_20260221_180613`
- micro accuracy: 48.61%
- macro task accuracy: 48.46%
- task coverage: 108/125
- observed episodes: 323
- teacher: glm-4.7 @ https://api.z.ai/api/coding/paas/v4

## Best coverage baseline

- run: `miniwob_train_20260221_164638`
- micro accuracy: 36.64%
- macro task accuracy: 36.64%
- task coverage: 125/125
- observed episodes: 625
- teacher: GUI_8B @ http://localhost:7999/v1

## Top runs by accuracy

| run | label | micro acc | tasks | episodes | action err | teacher |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `miniwob_train_zai_20260221_175937` | partial | 50.00% | 6/125 | 16 | 3.750 | glm-4.7 |
| `miniwob_train_zai_20260221_180613` | analysis_candidate | 48.61% | 108/125 | 323 | 1.260 | glm-4.7 |
| `miniwob_train_20260221_164638` | sft_candidate | 36.64% | 125/125 | 625 | 1.840 | GUI_8B |
| `miniwob_train_zai_20260224_182424` | partial | 0.00% | 1/125 | 1 | 0.000 | glm-4.7 |
| `miniwob_parallel_smoke2_20260221_161254` | broken | 0.00% | 0/2 | 0 | 0.000 | GUI_8B |
| `miniwob_parallel_smoke_20260221_161222` | broken | 0.00% | 0/2 | 0 | 0.000 | GUI_8B |
| `miniwob_portscan_test2_20260221_161902` | broken | 0.00% | 0/2 | 0 | 0.000 | GUI_8B |
| `miniwob_portscan_test_20260221_161702` | broken | 0.00% | 0/2 | 0 | 0.000 | GUI_8B |

## Worst tasks in the best-accuracy run

| task | success rate | episodes | avg reward | avg steps |
| --- | ---: | ---: | ---: | ---: |
| `ascending-numbers` | 0.00% | 3 | 0.000 | 10.000 |
| `bisect-angle` | 0.00% | 3 | 0.000 | 7.667 |
| `choose-date` | 0.00% | 3 | 0.000 | 10.000 |
| `choose-date-easy` | 0.00% | 3 | 0.000 | 10.000 |
| `choose-date-medium` | 0.00% | 3 | 0.000 | 10.000 |
| `choose-date-nodelay` | 0.00% | 3 | 0.000 | 10.000 |
| `circle-center` | 0.00% | 3 | 0.000 | 7.667 |
| `click-collapsible-2` | 0.00% | 3 | 0.000 | 7.667 |
| `click-collapsible-2-nodelay` | 0.00% | 3 | 0.000 | 5.333 |
| `click-color` | 0.00% | 3 | 0.000 | 7.000 |
| `click-link` | 0.00% | 3 | 0.000 | 7.000 |
| `click-pie` | 0.00% | 3 | 0.000 | 10.000 |

## Best tasks in the best-accuracy run

| task | success rate | episodes | avg reward | avg steps |
| --- | ---: | ---: | ---: | ---: |
| `use-spinner` | 100.00% | 3 | 1.000 | 2.333 |
| `use-autocomplete` | 100.00% | 3 | 1.000 | 2.667 |
| `unicode-test` | 100.00% | 3 | 1.000 | 1.000 |
| `simple-arithmetic` | 100.00% | 3 | 1.000 | 2.000 |
| `simple-algebra` | 100.00% | 3 | 1.000 | 2.000 |
| `scroll-text` | 100.00% | 3 | 1.000 | 2.333 |
| `phone-book` | 100.00% | 3 | 1.000 | 3.000 |
| `multi-layouts` | 100.00% | 3 | 1.000 | 4.000 |
| `identify-shape` | 100.00% | 3 | 1.000 | 1.000 |
| `generate-number` | 100.00% | 3 | 1.000 | 2.333 |
| `focus-text-2` | 100.00% | 3 | 1.000 | 1.000 |
| `focus-text` | 100.00% | 3 | 1.000 | 1.000 |

## Broken runs worth noting

- `miniwob_parallel_smoke2_20260221_161254` — episodes=0, teacher=GUI_8B @ http://localhost:7999/v1
- `miniwob_parallel_smoke_20260221_161222` — episodes=0, teacher=GUI_8B @ http://localhost:7999/v1
- `miniwob_portscan_test2_20260221_161902` — episodes=0, teacher=GUI_8B @ http://localhost:7999/v1
- `miniwob_portscan_test_20260221_161702` — episodes=0, teacher=GUI_8B @ http://localhost:7999/v1
- `miniwob_throughput_20260221_161201` — episodes=0, teacher=GUI_8B @ http://localhost:7999/v1
- `miniwob_train_20260221_161520` — episodes=0, teacher=GUI_8B @ http://localhost:7999/v1
- `miniwob_train_20260221_162050` — episodes=0, teacher=GUI_8B @ http://localhost:7999/v1
- `miniwob_train_fixcheck_20260221_164438` — episodes=0, teacher=GUI_8B @ http://localhost:7999/v1
- `miniwob_train_zai_20260221_172642` — episodes=0, teacher=glm-5 @ https://api.z.ai/api/coding/paas/v4
- `miniwob_train_zai_20260224_181259` — episodes=0, teacher=glm-4.7 @ https://api.z.ai/api/coding/paas/v4

## Takeaways

- Highest-accuracy substantial run is `miniwob_train_zai_20260221_180613` at 48.61%.
- Broadest successful task coverage is `miniwob_train_20260221_164638` covering 125/125 tasks.
- Existing data is already large enough to baseline quality before generating more traces.
- Recent failures are dominated by backend auth and connection issues, so backend stability should be fixed before scaling collection.
- The next high-leverage step is a failure taxonomy over the best successful runs, especially the worst-performing task families.
