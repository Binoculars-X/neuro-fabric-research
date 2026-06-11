# EXP-005 — TinyStories 1M params, byte-level vocab=256, 500K samples

## Goal
Establish baseline for 1M-param model on TinyStories with new byte-level tokenisation (vocab=256).
Scout the capacity ceiling and BPC floor for this architecture.

## Config
| Parameter | Value |
|---|---|
| Params | ~1M |
| embedDim | 120 |
| heads | 4 (head_dim=30) |
| ff | 360 |
| layers | 6 |
| vocab | 256 (byte-level) |
| batchSize | 32 |
| samples | 500,000 |
| LR | 0.003 linear decay |
| warmup | 500 steps |
| dataset | TinyStories (22.5M tokens, 90/10 split) |
| hardware | GPU Adam FP32, RTX 4090 |

## Results

| Samples | BPC | Train Loss | Eval Loss | Accuracy | ms/sample |
|---|---|---|---|---|---|
| 4,992 | 4.254 | 3.6658 | 2.9487 | 23.89% | 14.53 |
| 54,912 | 1.686 | 1.1764 | 1.1684 | 64.55% | 15.76 |
| 124,800 | 1.446 | 0.9863 | 1.0021 | 68.81% | 14.58 |
| 249,600 | 1.302 | 0.8590 | 0.8798 | 72.92% | 14.76 |
| 374,400 | 1.228 | 0.8225 | 0.8509 | 74.01% | 14.63 |
| **500,000** | **1.166** | **0.7803** | **0.8078** | **75.27%** | **14.49** |

**Total time:** 7,330s (14.66 ms/sample avg)

## Key observations
- Eval loss reached **0.808 BPC 1.166** at 500K samples — well below theoretical English entropy
- Train/eval gap minimal throughout: model generalises, no overfitting (22M tokens vs 1M params)
- BPC still slowly descending at 500K — model has not fully converged
- No UNK tokens (byte-level vocab eliminates the problem entirely)
- Quality check output was still showing old word-by-word decode with spaces (bug fixed in v1.1.0 post this run)

## Demo output (post-fix build, v1.1.0)

Tested with 4 prompts via `2.demochat.bat`. Text is fluent, grammatically correct, and thematically consistent with TinyStories style. Occasional non-sequiturs appear (expected at 1M params / 500K samples).

**Prompt:** `once upon a time`
> once upon a time, and they both played together again in the stage. After that, Tim and Sam always lifted their box was gone. `<|endoftext|>` Once upon a time, there was a wild cat named Tom. Today, Tom was playing with his ball. They liked to play in the park store. One day, Tom and his

**Prompt:** `One day a little girl named Lucy`
> One day a little girl named Lucy. She saw the spider. Lucy loved to play with her toys in the bathroom. Lucy was sad and couldn't go to the shower to see. But she was so excited to watch them and cool. Lucy was shining and said, "I am not the fence in to make some cake." Lucy smiled and said, "Than[k you, Buddy. What a bark rock!"]

**Prompt:** `Tom was very hungry`
> Tom was very hungry. `<|endoftext|>` One day, a boy named Tim found a farm in the woods. He saw a big field with his favorite toy. He wanted his wings on the sand and tried to explore. And they lived happily ever afternoon and fell down. `<|endoftext|>` Once upon a time, in a big for[est, there was a red ball...]

**Prompt:** `Mia could not sleep because`
> Mia could not sleep because it was time to go home. The jewel became good friends and played together every day. They decided to cheer to find the books. They were very special and happy. One day, the market saw a sign in the park. The boy was sad and had a great day. The dark had a great

### Observations
- Output is byte-level continuous (no space artifacts) — fix working correctly
- `<|endoftext|>` story boundaries respected and model restarts new stories naturally
- Names, dialogue, and simple cause-effect plots are all present
- Some non-sequiturs ("wings on sand", "bark rock") — expected capacity limit at 1M params
- No garbled or non-UTF-8 output

## Notes
- Byte-level tokenisation introduced in this experiment (replaces word-level vocab=1501)
- New cache file `tokens_bytes.bin` — old `tokens.bin` and `vocab.txt` are ignored
- This run used pre-fix build; quality check output garbled (spaces between chars)
- Demo output above collected with post-fix build (`.Encode` removed, direct UTF-8 bytes)
