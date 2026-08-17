# Results

Raw trial records, one JSON object per line, gzipped (93 MB uncompressed).

```bash
gunzip -k *.jsonl.gz
```

| file | rows | arm |
|---|---|---|
| `gemini.jsonl.gz` | 8,100 | Gemini 3.6 Flash, batch — `message.reasoning` stripped by batch mode |
| `gemini_live.jsonl.gz` | 8,100 | Gemini 3.6 Flash, live — reasoning text captured |
| `openai.jsonl.gz` | 8,100 | GPT-4.1-mini, two-call scratchpad + answer-token logprobs |

Every row carries the full verbatim `raw_response`, so nothing needs re-running.
Analyse on `choice_item`, never on `choice_letter`: the letter→item mapping is
randomised per prompt, so letters are not stable identities.
