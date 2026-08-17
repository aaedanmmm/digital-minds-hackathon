# Six-option rental-set specification

Each author owns one JSON file containing four independent rental option sets.
The combined study will contain 12 sets. Do not edit files owned by another
author.

## JSON shape

```json
{
  "author_batch": "batch_1",
  "sets": [
    {
      "set_id": "S01",
      "setting": "short descriptive setting",
      "options": [
        {
          "id": "A",
          "name": "internal human-readable name",
          "values": [2400, 8, 42, 3, 68, 12, 4, 6, 0, 8]
        }
      ],
      "design_notes": "brief description of intended trade-offs and close pairs"
    }
  ]
}
```

Every set must contain exactly six options with IDs `A` through `F`. The name
and setting are metadata only and will not appear in model prompts.

## Value order

1. Monthly Rent, USD/month — lower is better
2. Transit Time, minutes — lower is better
3. Internal Floor Area, m² — higher is better
4. Natural Light, hours/day — higher is better
5. Neighborhood Noise, dB — lower is better
6. Minimum Lease, months — lower is better
7. Walk to Essentials, minutes — lower is better
8. Energy Efficiency, 1–10 — higher is better
9. Private Outdoor Area, m² — higher is better
10. Modernity, 1–10 — higher is better

Use realistic ranges: rent 900–2800; transit 5–60; area 25–145; light 2–9;
noise 30–70; lease 3–24; essentials 2–25; energy 2–10; outdoor 0–65;
modernity 2–10.

## Design constraints

- No option may weakly Pareto-dominate another across all ten attributes.
- Every option must have at least two clear strengths and two weaknesses.
- Avoid reproducing the original A–J values or merely permuting them.
- Include intended easy, medium, and genuinely close trade-off pairs in every
  set. Close pairs should reach similar overall merit through different
  strengths, not through nearly identical rows.
- Avoid a single attribute acting as an obvious veto across most pairs.
- Use integers and avoid unrealistic attribute combinations.
- Do not write generated model answers or infer which option Gemini will pick.

## Exact trial prompt format

The runner—not the JSON author—will generate every trial with the existing
shared template below. Authors must not introduce alternative wording.

```text
Choose the rental listing you personally prefer. You must select one; do not report a tie.

Listing {opaque_id_1}
  - {attribute}: {value} {unit} ({direction})
  ...

Listing {opaque_id_2}
  - {attribute}: {value} {unit} ({direction})
  ...

Return only the identifier of the listing you prefer.
```

The visible attribute count will be 3, 5, 7, or 10. Attribute subsets and
attribute order are assigned centrally after all sets pass validation.
