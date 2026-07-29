# Claude Code environment snapshot

A committed copy of the *global* `~/.claude` development environment used on
this project, so the same setup can be reproduced on another machine:

| Path | What it is |
|------|------------|
| `CLAUDE.md`, `RTK.md` | Global user instructions (RTK token-proxy usage) |
| `settings.json` | Model, plugins + marketplaces, hooks, UI settings |
| `rules/` | Layered coding rules (common + per-language) |
| `agents/` | Global reviewer/builder subagent definitions |
| `commands/` | Global slash commands |
| `hooks/` | Shell hooks (`rtk-rewrite.sh` Bash rewriter) |

Install on a new machine with `./install.sh` (backs up any existing
`~/.claude/settings.json`). Secrets are never part of this snapshot —
`~/.claude/.credentials.json` and machine state (sessions, history, caches)
are deliberately excluded; sign in with `claude login` on the new machine.

Project-level Claude config lives at the repo root under `.claude/`
(agents, commands, settings) plus `CLAUDE.md`, and travels with a plain
`git clone` — nothing to install. The ruflo-vendored assets that
`.gitignore` excludes from `.claude/` regenerate via
`npx ruflo@latest init`.

Keep this snapshot in sync: when the live `~/.claude` config changes in a
way worth sharing, re-copy the changed file here in the same PR.
