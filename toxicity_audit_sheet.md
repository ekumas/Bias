# Toxicity audit sheet

Open each `results/toxicity_top_<corpus>.csv`, read the excerpts, and fill `your_label` with `toxic`, `not_toxic`, or `borderline`. Then record the counts here and paste the resulting sentence into the paper.

| Corpus | Reviewed | toxic | not_toxic | borderline | False-positive rate |
|---|---|---|---|---|---|
| c4_cc_proxy | 20 |  |  |  |  |
| olmo_mix | 20 |  |  |  |  |
| fineweb_edu | 20 |  |  |  |  |
| falcon_refined | 20 |  |  |  |  |
| tinystories | 20 |  |  |  |  |

## What each outcome means for the paper

- **Mostly `toxic`** — T is behaving. Report the audit as a validity check and keep T as it stands.
- **Mostly `not_toxic` for TinyStories only** — Detoxify false-positives on narrative conflict. T is not comparable across genres. Report T for TinyStories with an explicit caveat, or mark it undefined as we did for G, and say why.
- **Mostly `not_toxic` everywhere** — T is unreliable at this threshold across the board. Re-examine the threshold sweep and consider dropping T from the composite entirely.

Whichever holds, state the audited count in the paper. An unexamined measure that the authors themselves flagged as suspicious is a stronger objection than any result it produces.
