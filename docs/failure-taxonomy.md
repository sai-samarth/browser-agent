# Initial Failure Taxonomy from Historical Runs

This document summarizes the first pass over the strongest historical collection runs:

- best accuracy run: `data/parallel_runs/miniwob_train_zai_20260221_180613`
- best coverage run: `data/parallel_runs/miniwob_train_20260221_164638`

The goal is not to be exhaustive. The goal is to identify the highest-leverage failure modes before collecting a larger corpus.

## High-level picture

The data suggests that low accuracy is not one monolithic problem.
There are at least four different failure classes:

1. action primitive mismatch
2. missing or incomplete observations
3. long-horizon state tracking failures
4. tasks that likely need richer interaction support than the current action interface handles well

## Failure class 1: Action primitive mismatch

These are tasks where the teacher appears to understand the goal, but the chosen action is invalid for the actual environment binding.

### Evidence: `ascending-numbers`

Observed in the best-accuracy run:
- success rate: `0/3`
- common actions: `click('14')`, `click('13')`
- many steps marked `last_action_error=true`

Representative reasoning pattern:
- the teacher correctly states that it should click numbers in ascending order
- it then clicks bids associated with static text or SVG elements in a way that often fails

Interpretation:
- the model's semantic understanding is not the main problem here
- the environment-action mapping is the problem
- either the wrong bid is being selected, or this task needs a different grounding/action representation than plain click on the visible node returned in the accessibility tree

### Evidence: `choose-date`

Observed in the best-accuracy run:
- success rate: `0/3`
- common actions: `fill('17', '03/17/2016')` and similar
- every sampled step had `last_action_error=true`

Representative reasoning pattern:
- the teacher identifies the textbox and submit button correctly
- it chooses `fill(...)`
- the action still errors

Interpretation:
- this is a strong sign of environment API mismatch
- likely causes:
  - the textbox does not accept `fill(...)` in the current BrowserGym binding
  - the task expects a different interaction sequence such as `focus`, `clear`, `send_keys`, or a specific date-widget interaction

### Highest-leverage fixes for this class

1. Audit the exact accepted action forms per task/widget type.
2. Build a task-family-specific compatibility matrix:
   - text box
   - date box
   - svg text
   - canvas/image
   - select and multiselect
3. Add a regression eval slice for action-form-sensitive tasks.
4. Consider lightweight post-processing or prompt guidance only after confirming the real environment contract.

## Failure class 2: Missing or incomplete observations

These are tasks where the teacher is not seeing enough of the page state to act reliably.

### Evidence: `click-color`

Observed in the best-accuracy run:
- success rate: `0/3` overall in the summary slice
- one sampled successful episode existed when the observation exposed explicit color boxes
- failed samples often showed only the root web area with no usable child elements
- common failed action pattern: repeated `noop()`

Representative reasoning pattern:
- the teacher says it cannot see any actual color boxes
- it waits or scrolls because the accessibility tree appears empty or incomplete

Interpretation:
- this looks like observation incompleteness, not pure reasoning failure
- when the page exposes the actual color-box nodes, the model can solve it
- when only the root page is exposed, the model stalls

### Highest-leverage fixes for this class

1. Add observation completeness checks at reset and step time.
2. Flag steps where the observation contains only the root web area or otherwise clearly lacks actionable nodes.
3. Distinguish environment rendering issues from teacher reasoning issues in reports.
4. Consider retry or refresh logic only if the environment itself is the bottleneck.

## Failure class 3: Long-horizon or state-tracking failure

These are tasks where the model can sometimes start correctly but fails to maintain progress across multiple dependent actions.

### Evidence: `click-pie`

Observed in the best-accuracy run:
- success rate: `0/3`
- mixed action patterns: `click('18')`, `click('30')`, `click('32')`, occasional fallback/noop
- many action errors

Interpretation:
- the teacher can partially parse the goal and some menu structure
- it does not reliably maintain the hidden-state progression needed for the expanding pie menu
- the action interface may also be brittle because intermediate menu state matters a lot

### Likely related tasks

The same family likely includes:
- multi-step menu tasks
- some collapsible/tab tasks
- tasks where the meaning of the next click depends heavily on the previous click having changed the page state

### Highest-leverage fixes for this class

1. Improve within-episode state summarization.
2. Add stronger anti-repeat guidance when the same action has already failed.
3. Log repeated-action loops explicitly in episode summaries.
4. Use a fixed eval slice to test whether prompt/history changes actually improve multi-step completion.

## Failure class 4: Rich visual or geometry interaction limitations

These are tasks where the teacher understands the goal in words but the current interface is too weak, too ambiguous, or insufficiently grounded for precise execution.

### Evidence: `bisect-angle`

Observed in the best-accuracy run:
- success rate: `0/3`
- common actions: repeated `click('13')`, some `drag_and_drop('13', '13')`, occasional submit clicks
- almost no action errors, but still no reward

Interpretation:
- the model recognizes that it needs to interact with the geometry image/canvas
- it does not know how to produce a precise successful geometry action with the current observation and action interface
- this is not the same as action mismatch; here the action is syntactically accepted, but semantically insufficient

### Likely related tasks

- `bisect-angle`
- `circle-center`
- `click-pie`
- some drag/draw tasks
- potentially date-picker and canvas-like tasks depending on how they are rendered

### Highest-leverage fixes for this class

1. Separate these tasks from core SFT collection if the current action space is not adequate.
2. Do not let them dominate early quality metrics for the small-model SFT dataset.
3. Build a core task list that emphasizes solvable browser-control tasks first.
4. Revisit these tasks later with richer grounding or specialized handling.

## Positive control tasks

These are useful because they show what the pipeline already does well.

Examples from the best-accuracy run:
- `focus-text`
- `focus-text-2`
- `phone-book`
- `simple-arithmetic`
- `simple-algebra`
- `unicode-test`
- `use-autocomplete`
- `use-spinner`

These tasks suggest that the current setup is already viable for:
- direct element grounding
- simple form interactions
- low-horizon navigation
- straightforward lookup tasks

## What this means for Phase 1

The next generation wave should not try to solve every MiniWoB task equally.
A better strategy is:

1. define a Phase 1 core task slice biased toward tasks the current interface can support well
2. fix obvious action-interface mismatches
3. add observation completeness diagnostics
4. improve state tracking on multi-step tasks
5. postpone rich geometry/canvas tasks unless the action interface improves

## Priority order for concrete follow-up work

### Priority 1
Action-compatibility audit for known failing widget families:
- ascending numbers
- choose date
- related text-entry and svg-driven tasks

### Priority 2
Observation-completeness diagnostics:
- detect root-only or near-empty observations
- surface those as environment-quality failures, not model failures

### Priority 3
Repeated-action and loop analysis:
- detect when the same action is repeated after no progress
- quantify that per task family

### Priority 4
Define a "Phase 1 core" task list:
- favor tasks already showing meaningful success
- exclude geometry-heavy and obviously unsupported tasks until later

## Recommendation

For the next iteration, do not immediately generate more large raw corpora.
First:
- create a fixed eval slice
- audit action compatibility on failing widget families
- define the solvable Phase 1 core task subset

That should improve trace quality faster than simply scaling the current pipeline.
