# Codebook: tells coded by the model coders (v1, frozen Nov 7, 2026 after a 20-text pilot)

You will read short texts. For each text, count how many times each pattern below occurs. Count only what the definition says; do not guess who wrote the text. Text inside quotation marks that quotes someone else still counts.

Return, for each text, one integer per code (0 if absent).

| Code | Pattern | Counts | Does not count |
| --- | --- | --- | --- |
| contrast | A contrast frame that denies one description and asserts another: "It's not X, it's Y", "This isn't about X. It's about Y.", "not just X, but Y", "not X, but Y" | "It's not a bug, it's a feature." "Not just faster, but cheaper." "The problem isn't the code. It's the process." | "not only X but also Y"; a plain negation with "but" that adds a separate point ("I'm not sure, but I think..."; "It doesn't work, but I'll try again"); the reversed order "Y, not X" ("a sign of care, not panic"); "less X, more Y"; a denial followed by a different suggestion in the next sentence unless that sentence restates the subject with "it's", "it is", "this is", "that's" or "they're" ("You don't need one approach. You can mix them." does not count) |
| rather_than | The phrase "rather than" | each occurrence | "would rather" |
| tricolon | A list of exactly three coordinated items joined by "and" or "or" (words, phrases or clauses) | "fast, cheap and reliable"; "we planned, built, and shipped it" | lists of two or of four or more; three items without "and"/"or" |
| staccato | A run of two or more consecutive very short sentences (five words or fewer each) in one paragraph; count runs, not sentences | "No setup. No config. Just results." (one run) | a single short sentence; list items; headings |
| signpost | A phrase that announces what is coming instead of saying it: "here's the thing", "here's why", "the key insight", "the real question is", "the bottom line", "let's dive in", "let that sink in", "the truth is", "make no mistake", "spoiler:" | each occurrence | the same words used literally ("the bottom line of the page") |
| closer | The text ends with a summary closer: the last paragraph, or one of the last two sentences, starts with "In conclusion", "In summary", "To sum up", "Overall,", "Ultimately,", "In short", "All in all", "In the end", "The takeaway", "Key takeaway", "TL;DR" | 1 if present, else 0 | "overall" used as an adjective ("the overall cost") |
| worth_noting | "it's worth noting", "worth mentioning", "worth pointing out", "it's important to note/remember/understand/keep in mind", or a sentence starting "Note that" | each occurrence | |
| participial | A present participle (-ing verb) heading an adverbial clause that hangs off a main clause, set off by a comma or opening the sentence: "..., highlighting the need for change"; "Looking at the data, we see..." | each clause | -ing words that are nouns ("the meeting"), adjectives ("an interesting idea"), progressive verbs ("we are building"), gerund subjects with no comma ("Using X creates confusion"), or clauses with "while/when/by/after/before" in front |
| self_answer | A question immediately answered by the writer in a short sentence of four words or fewer: "Does it work? Absolutely." | each occurrence | questions not answered right away |
| opener | The text opens with a chatbot-style opener: "Great question", "Good question", "You're (absolutely) right", "Absolutely", "Certainly", "Of course", "I understand", "That's a great/fair point", "Totally", "Sure" | 1 if present, else 0 | anything not on the list ("Agreed", "Very true") |
| hedge_closer | The last sentence starts with "However", "That said", "That being said", "Of course" or "Still," | 1 if present, else 0 | |

## Content codes (separate pass, 0 or 1)

| Code | 1 when | 0 when |
| --- | --- | --- |
| generic | The text makes no concrete, checkable or first-hand point: no specific example, number, name, experience or detail that ties it to this situation. It could be pasted under many similar questions | It contains at least one concrete specific (an example, a number, a named thing, a personal experience, a precise step) |
| hype | The text inflates its subject: grand or emotive framing ("game-changer", "revolutionary", "a testament to"), every claim emphasized, or a motivational tone out of proportion to the content | Plain, proportionate tone, even if enthusiastic about one specific thing |
