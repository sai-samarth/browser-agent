# Action Compatibility Audit - Initial Pass

This document records the first concrete audit of task families where the teacher's reasoning appears plausible but the action interface or environment binding may be the real bottleneck.

This is intentionally operational.
The purpose is to decide what to test next, not to claim final root cause proof for every task.

## Summary matrix

| Task family | Representative task | Observed teacher behavior | Likely issue type | Priority |
| --- | --- | --- | --- | --- |
| SVG/static-text clicking | `ascending-numbers` | clicks visible number bids, many action errors | click target binding mismatch | high |
| Date input widgets | `choose-date` | chooses `fill(...)`, action errors every time | unsupported widget action sequence | high |
| Incomplete tree / missing children | `click-color` | repeated `noop()` when only root area is visible | observation incompleteness | high |
| Hidden-state menu interaction | `click-pie` | mixed clicks, occasional fallback, poor progress | long-horizon plus interaction fragility | medium |
| Geometry / canvas interaction | `bisect-angle` | repeated accepted actions with no reward | semantically insufficient action space | medium |

## 1. SVG and static-text grounded tasks

### Representative task: `ascending-numbers`

Observed behavior:
- teacher correctly explains the objective: click 1 through 5 in ascending order
- common actions include `click('14')` and `click('13')`
- many steps are marked `last_action_error=true`

What this suggests:
- the teacher is not fundamentally misunderstanding the task
- the problem is likely that the bid exposed in the tree is not the correct actionable target
- the task may expose text nodes while requiring clicks on a different underlying shape or container

Hypotheses to test:
1. The visible numeric text bid is not clickable, but a parent SVG/container bid is.
2. The correct action may need a different target resolution strategy than the current prompt encourages.
3. The accessibility tree may be exposing the wrong granularity for this task family.

Recommended experiments:
- inspect raw observations for the same seeds and compare successful vs failing bids if any exist
- try parent-container targeting if present in the tree
- log whether the clicked bid corresponds to `StaticText`, `graphics-symbol`, `SvgRoot`, or another node type

## 2. Date and structured input widgets

### Representative task: `choose-date`

Observed behavior:
- the teacher identifies the textbox and submit button correctly
- it repeatedly emits actions like `fill('17', '03/17/2016')`
- sampled steps show `last_action_error=true` consistently

What this suggests:
- the teacher's semantic parse is probably fine
- `fill(...)` is likely not the right primitive for this widget in the current environment
- the widget may require a sequence like focus -> clear -> send_keys, or it may expose a date picker rather than a normal text box

Hypotheses to test:
1. `fill(...)` is rejected on this widget while `focus(...)` and typing-style actions work.
2. The date field is not a normal textbox despite appearing as one in the observation.
3. The task expects submission through a specific interaction order.

Recommended experiments:
- manually reproduce one seed and test candidate action sequences against the real env
- compare `fill`, `focus`, `clear`, and any available key-entry primitives
- create a widget-family compatibility table for text-like controls

## 3. Observation incompleteness

### Representative task: `click-color`

Observed behavior:
- failed episodes often show only the root page area in the observation
- the teacher responds with `noop()` or scrolls because it cannot see actual colored boxes
- when the boxes are visible in the observation, the teacher can solve the task

What this suggests:
- the teacher is not always the limiting factor
- the environment or observation pipeline is sometimes exposing an incomplete tree
- these episodes should be tagged separately from genuine reasoning failures

Hypotheses to test:
1. Some resets produce incomplete initial observations.
2. The page may require one extra tick before the action tree is fully available.
3. Observation truncation or missing fields may hide the actionable nodes.

Recommended experiments:
- add observation completeness diagnostics to every step row
- count how often root-only or near-empty observations occur by task
- compare success rate conditioned on observation completeness

## 4. Long-horizon hidden-state interactions

### Representative task: `click-pie`

Observed behavior:
- action sequences vary across episodes
- some episodes use sensible opening clicks
- some later steps fall back or drift
- many runs do not convert early progress into reward

What this suggests:
- the task likely requires stronger within-episode state handling
- the action space may be good enough in principle, but not with the current prompt/history discipline
- this is less of a raw binding mismatch than `choose-date` or `ascending-numbers`

Hypotheses to test:
1. The teacher loses track of whether the menu is already expanded.
2. Repeating the same click after no progress should be discouraged more strongly.
3. A shorter and more state-focused history window may work better than verbose generic reasoning.

Recommended experiments:
- detect repeated-action loops at episode level
- compare success on the same task under different history-window settings
- add explicit no-progress summaries between steps

## 5. Geometry and canvas-style interactions

### Representative task: `bisect-angle`

Observed behavior:
- actions like `click('13')` and `drag_and_drop('13', '13')` are accepted
- very few action errors occur
- reward still stays at zero

What this suggests:
- syntax is accepted but semantics are insufficient
- the teacher knows it must interact with the geometry canvas, but the current interface does not give it enough grounding to do so precisely

Hypotheses to test:
1. The task needs coordinate-aware control or richer visual grounding.
2. The current accessible-tree representation is too weak for precise geometric actions.
3. These tasks should be excluded from the early SFT corpus until a richer action interface exists.

Recommended experiments:
- do not prioritize these tasks for Phase 1 corpus generation
- isolate them into a separate research bucket rather than mixing them with core SFT tasks

## Immediate follow-up tasks

### High priority
1. Build a widget/action compatibility matrix for:
   - text inputs
   - date inputs
   - svg/static-text click targets
2. Add collector diagnostics for:
   - root-only observations
   - action loops
   - repeated no-progress steps
3. Separate environment-quality failures from model-quality failures in summaries

### Medium priority
4. Introduce a fixed eval slice covering one task from each failure class
5. Re-test prompt/history settings only after action compatibility is better understood

## Recommendation

The first collector fixes should target:
- observation completeness diagnostics
- repeated-action loop diagnostics
- action-family auditing support

Those changes are likely to improve corpus quality faster than trying to brute-force teacher prompting alone.
