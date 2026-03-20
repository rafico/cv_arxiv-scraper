# Product Roadmap

This roadmap reframes the project as a local research triage product, not just a scraper.

## P0: Activation And Daily Habit

### Goal
Make first-time users reach value quickly and make repeat usage feel obvious.

### Problems
- Setup trust is weakened by documentation and workflow inconsistencies.
- The product opens on a paper grid before explaining the user journey.
- `Save` exists as an action but not as a clear destination or workflow.

### User Stories
- As a first-time user, I want to understand what to do first so I can get useful results quickly.
- As a daily user, I want a clear inbox and a clear saved queue so I can triage without losing papers.
- As a user, I want feedback actions to behave consistently so I trust the ranking loop.

### Suggested UI Changes
- Add a first-run dashboard state with a short 3-step setup sequence.
- Split the main workflow into `Inbox` and `Saved`.
- Rename `Skip` in the UI to `Not Interested`.
- Add status cards for saved count, last scrape, and tracked interests.
- Add a lightweight `Why it matched` explanation on every card.

### Success Signals
- User can complete first scrape without reading source code.
- User can find previously saved papers in one click.
- User can explain the difference between `Save` and `Not Interested`.

## P1: Control And Explainability

### Goal
Help users tune ranking instead of treating the ranking model as a black box.

### Problems
- Ranking logic exists in code but is barely visible in the UI.
- Whitelists are powerful but feel like raw configuration, not product controls.
- Metadata already collected by the scraper is underused in the dashboard.

### User Stories
- As a user, I want to know why a paper ranked highly so I can decide whether to trust it.
- As a user, I want to tune authors, labs, topics, and freshness without editing YAML mentally.
- As a user, I want code, dataset, demo, DOI, and category signals on the paper card.

### Suggested UI Changes
- Add a score explanation drawer or hover details.
- Add settings for boost, mute, and freshness preferences.
- Show categories and resource badges on the dashboard by default.
- Add filters for saved papers, categories, and resource availability.

### Success Signals
- Users can answer “why am I seeing this?” from the UI alone.
- Users can improve results after one settings pass.
- Paper cards reduce click-through just to discover basic context.

## P2: Automation And Compounding Value

### Goal
Make the product work for users even when they do not open it every day.

### Problems
- Gmail digest and AI settings are useful but feel like advanced setup chores.
- Scrape history exists in the data model but is not presented as a reliability surface.
- There is no explicit notion of digest quality or saved-paper follow-up.

### User Stories
- As a busy user, I want the digest to feel curated, not just exported.
- As a user, I want to trust that daily automation is healthy.
- As a user, I want the app to learn over time from what I save and ignore.

### Suggested UI Changes
- Add automation health and last-send status to the dashboard/settings.
- Add digest preview with subject line and paper count before sending.
- Add scrape history and recent run outcomes.
- Add lightweight follow recommendations such as `Follow author` or `Mute topic`.

### Success Signals
- Users can tell whether automation is healthy in under 10 seconds.
- Digest setup feels optional and secondary, not required for core value.
- The product shows evidence that feedback improves future ranking.

## Implementation Order

1. Ship the P0 dashboard workflow improvements first.
2. Expose ranking explanations and metadata next.
3. Improve automation visibility only after the core triage loop feels solid.
