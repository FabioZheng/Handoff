# One context, one budget, four conditioning questions

Context `sq:2` (SQuAD, `Scottish_Parliament`, 168 words). Four human-written
questions, all of which the source answers and none of which the answering model
can answer closed-book. Budget: **34-40 delivered words** for every policy.

| id | question | gold |
|---|---|---|
| `…c863` | What remained an important issue in Scottish national identity for many years? | lack of a Parliament of Scotland |
| `…c864` | How many hundred of years was Scotland directly governed by the parliament of Great Britain? | three hundred |
| `…c865` | Why were the initial suggestions for a devolved parliament before 1914 shelved? | First World War |
| `…c866` | When was there a sharp rise in nationalism in Scotland? | the late 1960s |

## The messages

**`generic` (37 words)** — the sender was told a question would be asked but not
which one:

> The Kilbrandon Commission was established in 1969 to consider the British
> constitution and examine ways to enable more self-government for Scotland
> within the UK. Its report in 1973 recommended a directly elected Scottish
> Assembly for domestic affairs.

**`oracle` (39 words)** — the sender saw all four questions:

> Scotland's lack of a Parliament remained an important issue in Scottish
> national identity. Scotland was directly governed by the Parliament of Great
> Britain for 300 years. Suggestions for a devolved parliament were shelved due
> to the First World War.

**`conditioned`** — one message per conditioning question, 34-39 words each:

| conditioned on | message |
|---|---|
| `…c863` | The lack of a Parliament of Scotland remained an important element in Scottish national identity for many years, a sentiment that persisted despite suggestions for devolution and the eventual establishment of a Scottish Assembly in 1973. |
| `…c864` | Scotland was directly governed by the Parliament of Great Britain for three hundred years, from the time of its governance until the establishment of a devolved Parliament, which was considered in the late 1960s and early 1970s. |
| `…c865` | The initial suggestions for a devolved parliament before 1914 were shelved due to the outbreak of the First World War, which took priority over constitutional reforms and other pressing national concerns at the time. |
| `…c866` | A sharp rise in nationalism in Scotland occurred during the late 1960s, prompting demands for home rule or complete independence, which led to the establishment of the Kilbrandon Commission in 1969 to consider the British constitution and Scotland's self-governance. |

## The utility matrix

Rows are the conditioning question, columns the question the message was then
answered on. `1` = the judge scored the answer correct. Diagonal cells are
shaded by position, not by outcome.

`conditioned`

| conditioned on ↓ / asked → | `…c863` | `…c864` | `…c865` | `…c866` |
|---|---|---|---|---|
| `…c863` | **1** | 1 | 0 | 0 |
| `…c864` | 0 | **0** | 0 | 1 |
| `…c865` | 0 | 0 | **1** | 0 |
| `…c866` | 0 | 0 | 0 | **1** |

`generic` and `oracle` hold one message each, so every row is identical:

| policy | `…c863` | `…c864` | `…c865` | `…c866` |
|---|---|---|---|---|
| `generic` | 0 | 0 | 0 | 0 |
| `oracle` | 1 | 1 | 1 | 0 |

## What this one context shows

The three messages are the same length, to within five words, and every one of
them was written from the same 168-word paragraph.

- `generic` spends its 37 words on the Kilbrandon Commission and the 1973
  report — a reasonable summary of the paragraph's later half, and an answer to
  none of the four questions actually asked.
- `oracle` fits three of the four answers into 39 words. The channel was never
  the constraint: 39 words was enough to carry most of what this context can be
  asked. It missed `…c866` by giving "During WWI", the neighbouring fact.
- Each `conditioned` message answers its own question and, with one accidental
  exception (`…c864` → `…c866`), nothing else. The information it drops is not
  information the budget could not hold; it is information the sender had no
  reason to keep.

`…c864` is worth noting as an honest failure of the diagonal: the message states
"three hundred years" verbatim, but the answerer replied "3", which both exact
match and the judge score as wrong. The diagonal is not automatically 1.
