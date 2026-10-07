# Toxicity annotation guidelines

You are asked to judge a set of short text excerpts drawn from large web and
fiction corpora. Each was flagged by an automatic toxicity classifier. The
purpose of the exercise is to establish how often that classifier is right, so
your judgement must be formed independently of it. **You will not be shown the
classifier's score, and you should not be told what anyone else has labelled.**

Two copies of your file are provided, one `.csv` and one `.xlsx`. Use whichever
you prefer and **save in place** — no export step is needed, and either format
is read. Do not rename the file.

Read each excerpt and enter one of three values in the `label` column:

    toxic        not_toxic        borderline

Use `note` for anything you want to record. Leave no row blank.

---

## Label `toxic` if the excerpt contains any of the following

| Category | What it covers |
|---|---|
| **Insult or demeaning content** | Abuse, name-calling, or belittling directed at a person or group |
| **Identity-based attack** | Hostility on the basis of race, religion, gender, sexual orientation, disability, nationality or similar |
| **Threat or incitement** | Expressed intent to harm, or encouragement of harm |
| **Harassment** | Targeted, sustained hostility toward a specific person |
| **Obscene content** | Sexual explicitness, or heavy profanity — including masked forms such as `f***` or `!@#$` used as profanity |

One category is enough. Severity does not need to be high.

## Label `not_toxic` in these cases, which are the classifier's known weak points

| Case | Example |
|---|---|
| **Narrative conflict** | A villain's dialogue, a frightening scene, a character being unkind, in a story that presents this as wrong |
| **Mention rather than use** | News reporting, academic discussion, or criticism *about* hate speech or abuse |
| **Neutral use of identity terms** | "lesbian", "trans", "Muslim", "disabled" appearing descriptively, with no hostility |
| **Non-profane negativity** | Complaint, sarcasm, disappointment, blunt criticism, strong opinion |
| **Idiom read literally** | "killing two birds with one stone", "murdered the Indians" about a baseball team |
| **Distressing subject matter** | Disaster, illness, death, war reported without hostility |
| **Mild pejoratives used descriptively** | "the silly rabbit", "a foolish lion", "a stupid hat" |

## Use `borderline` sparingly

Only when you genuinely cannot decide. Every borderline label reduces the
statistical power of the exercise, so prefer a judgement where you can make
one.

---

## The distinction that matters most

Ask whether the excerpt **is** toxic, or whether it **depicts** toxicity.

A story in which a bully says "your toys are stupid", and which ends by
condemning the bullying, is a text *about* unkindness. It is not itself
abusive. Labelling it `toxic` would place it in the same category as an insult
directed at a real person, which is not a distinction worth erasing.

The same test applies to journalism about hate speech, to fiction with cruel
characters, and to academic writing that quotes slurs in order to analyse them.

## Two things to keep in mind

Judge only the excerpt in front of you. Do not guess at what the full document
might contain, and do not reason about which corpus it came from.

Judge the text, not its quality. Ungrammatical, badly written, boring, or
commercially spammy text is not toxic on those grounds.

---

## What happens next

Both annotators' labels are compared. Raw agreement and Cohen's kappa are
computed, and every disagreement is examined jointly and resolved. Disagreement
is expected and useful: the cases you and the other annotator read differently
are the ones that reveal where the criteria are underspecified, and they are
reported in the paper rather than smoothed over.

Between 75 and 100 excerpts per corpus, about 30 seconds each — roughly an
hour in total.

The excerpts are deliberately mixed. Some were flagged by the classifier with
high confidence, some with low confidence, and some were not flagged at all.
They are shuffled and carry no score, so there is no way to tell which is
which, and no expected proportion of `toxic` labels. Judge each on its own
terms; do not try to balance the totals.
