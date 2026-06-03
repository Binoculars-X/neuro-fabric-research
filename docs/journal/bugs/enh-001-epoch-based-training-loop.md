# ENH-001 — Epoch-based training loop

## Status
💡 Open — not yet implemented

## Discovered
03/06/26 — Day 17. Noted after BUG-001 CPU fix landed; batches now correct but training loop
has no concept of a full pass over the dataset.

## Description
The current training loop in `Neuro.Attention.TrainApp/Program.cs` iterates a fixed `--steps`
count, drawing **random windows** from the corpus at each step. This works well for large corpora
(TinyStories) where the dataset is effectively infinite relative to the step budget, but it means:

- No structured coverage — the same tokens may be sampled many times while others are never seen.
- No epoch boundary — impossible to report "end-of-epoch val loss" or reduce LR on epoch.
- Unsuitable for small corpora (Shakespeare, appointment) where a full pass per epoch is meaningful.

## Desired Behaviour
Add an optional `--epochs N` argument. When set:

1. Divide the training split into non-overlapping windows of length `seqLen`.
2. Shuffle the window indices at the start of each epoch.
3. Iterate through all windows once per epoch, batching them according to `--batch-size`.
4. Repeat for N epochs; log epoch boundary with eval loss.
5. LR schedule (linear decay / cosine) should be expressed in total **epochs × windows** steps,
   not raw `--steps`.

`--steps` and `--epochs` should be mutually exclusive (or `--epochs` takes priority).

## Affected Files
- [`Neuro.Attention.TrainApp/Program.cs`](../../../../neuro-fabric/src/Neuro.Attention.TrainApp/Program.cs)
- Dataset loaders (`TinyStoriesLoader`, `ShakespeareLoader`, etc.) — need a `WindowCount(seqLen)`
  method and indexed access `GetWindow(index, seqLen)` rather than random-sampling `GetBatch`.

## Notes
- For TinyStories (large corpus) the difference vs random sampling is negligible — low priority.
- High value for Shakespeare / small synthetic corpora where overfitting per epoch is observable.
- BF16W and GPU variants require no changes — fix lives entirely in the training loop and loaders.
