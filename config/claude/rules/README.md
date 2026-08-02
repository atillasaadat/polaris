# Rules

Layered coding rules installed into `~/.claude/rules/` by `../install.sh`.

## Structure

A **common** layer plus one directory per language this project actually uses:

```
rules/
├── common/          # Language-agnostic principles (always installed)
│   ├── coding-style.md   git-workflow.md   development-workflow.md
│   ├── testing.md        performance.md    patterns.md
│   └── hooks.md          agents.md         security.md
├── cpp/             # C++ — lib/, sim/, flight/
└── python/          # Python — tools/, tests/tools/, and the planned analysis/
```

- **common/** holds universal principles, with no language-specific code examples.
- **Language directories** extend the common rules with the language's tools, idioms
  and patterns. Each file opens with a pointer to its `../common/` counterpart.

Polaris is a C++/F´ and Python repo, so those are the only two language layers kept.
The upstream rule collection also ships `csharp`, `golang`, `java`, `kotlin`, `perl`,
`php`, `rust`, `swift` and `typescript` sets; they were dropped from this snapshot in
Push 50 because carrying rules for languages the repo does not contain is nine more
files to keep in sync and nothing to check them against. Re-add one from upstream if a
component ever needs it.

## Installation

`../install.sh` copies `rules/` wholesale into `~/.claude/rules/`, replacing whatever
is there so removals propagate. There is no per-language selection flag — the set
committed here *is* the set installed.

> Copy entire directories if you install by hand — do **not** flatten with `/*`.
> `common/` and the language directories contain files with the same names, so
> flattening lets a language file overwrite a common rule, and it breaks the relative
> `../common/` links the language files use.

## Rules vs skills

**Rules** define standards and checklists that apply broadly ("80% test coverage", "no
hardcoded secrets"). **Skills** provide deep reference material for a specific task.
Rules say *what*; skills say *how*.

## Rule priority

Language-specific rules take precedence over `common/` where they conflict (specific
overrides general — the same precedence model as `.gitignore`). Common rules that a
language layer may legitimately override are marked:

> **Language note**: This rule may be overridden by language-specific rules for
> languages where this pattern is not idiomatic.

For example `common/coding-style.md` makes immutability the default; a language whose
idiom is in-place mutation can say so in its own `coding-style.md` and win.

## Adding a language layer

1. Create `rules/<lang>/`.
2. Add the files that extend the common rules: `coding-style.md`, `testing.md`,
   `patterns.md`, `hooks.md`, `security.md`.
3. Open each with `> This file extends [common/xxx.md](../common/xxx.md) with <Language>
   specific content.`
4. Update the tree above, and note it in `config/claude/README.md`.
