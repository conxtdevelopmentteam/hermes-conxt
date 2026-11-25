# Conxt memory for Hermes Agent

**Long-term memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent) that a team can trust.**
Hermes remembers what you decided across sessions, and nothing reaches your team's memory
until a person approves it.

```bash
hermes plugins install conxt
hermes memory setup        # choose "conxt" and paste your Conxt API key
```

Get a key from the **Hermes** card at [conxt.dev/connect-ide](https://conxt.dev/connect-ide).
Any Conxt account can create one, Free included.

---

## Why Conxt

| | |
|---|---|
| 🧠 **Remembers for you** | Before each turn, Hermes gets back the decisions, preferences and project facts that match what you asked. |
| ✅ **A person approves team memory** | The agent can only *propose* a team memory. Nobody recalls it until someone on the team approves it in Conxt, and the plugin's key cannot approve its own proposals. |
| 🧾 **Every read is on record** | Each recall is logged in Conxt with what was read and who read it. |
| 🔁 **One memory across your tools** | The same memory serves Claude, ChatGPT, Cursor, Claude Code and Codex, so what Hermes learns isn't locked inside Hermes. |

## How it works

```
  your message ──► recall ──► Conxt ──► matching memories added as context
                                 ▲
  the conversation ──► batched capture ──► Conxt extracts memories from it
```

| When | What happens |
|---|---|
| **Before each turn** | Your message goes to Conxt and the matching memories come back: your own, plus your team's approved ones if a team is set. Greetings and one-word replies are skipped. |
| **After the conversation** | Turns are kept locally and sent in batches: every 5 exchanges, before Hermes compresses the context, when you switch or end a session, and at shutdown. Conxt extracts memories from each batch. |
| **When Hermes's built-in memory adds an entry** | The same text is saved to your Conxt memory. |
| **When the agent chooses** | `conxt_search` looks something up, `conxt_remember` saves a personal memory, and `conxt_propose` proposes a team memory (only when a team is set). |

**Safe for unattended runs.** Scheduled (cron), subagent and flush runs are never written to
automatically. If Conxt is slow or unreachable, recall is skipped for that turn. Unsent turns
wait for the next attempt, up to 400 messages, after which the oldest are dropped. Nothing
ever waits for a person, so an unattended run cannot hang.

## Setup

**Requirements:** Hermes Agent 0.21.6 or later, and a Conxt API key. Team memory needs the
Conxt Team plan.

Each tool gets its own key at [conxt.dev/connect-ide](https://conxt.dev/connect-ide), so you
can see what Hermes reads and revoke its key without touching your other tools.

From the [Hermes plugin catalog](https://github.com/NousResearch/hermes-agent/tree/main/plugin-catalog):

```bash
hermes plugins install conxt
hermes memory setup        # choose "conxt", paste your API key, optionally a team id
```

Or run the latest commit instead of the catalog's pinned version:

```bash
git clone https://github.com/conxtdevelopmentteam/hermes-conxt "$HERMES_HOME/plugins/conxt"
hermes memory setup
```

You can also choose it in `config.yaml`:

```yaml
memory:
  provider: conxt
```

### Configuration

| Setting | Where | Default | Meaning |
|---|---|---|---|
| `CONXT_API_KEY` | secret (`.env`, via `hermes memory setup`) | — | **Required.** Read through Hermes's scoped secrets, per profile. |
| `CONXT_TEAM_ID` | env, or `team_id` in `$HERMES_HOME/conxt.json` | none | A Conxt team id. Adds the team's approved memories to recall and turns on `conxt_propose`. |
| `CONXT_BASE_URL` | env, or `base_url` in `$HERMES_HOME/conxt.json` | `https://mcp.conxt.dev` | The Conxt API. Must be `https`. |

Each Hermes profile has its own key, team and settings.

## What leaves your machine

Everything goes to the Conxt API at `https://mcp.conxt.dev` (or your `CONXT_BASE_URL`),
authenticated with your API key, and to nothing else.

| What | What is sent |
|---|---|
| **Recall** | The text of your message (first 500 characters), and the team id if set. |
| **Capture** | The user and assistant messages, up to 10,000 characters each. Tool calls and tool results are not sent, but anything the assistant repeats in its reply is, such as file paths, command output or code it quotes. Conxt redacts secrets it recognises before storing memories. |
| **Saves and proposals** | The memory's text, its type and a title. |
| **Mirrored memory writes** | The text of each entry Hermes's built-in memory adds. |

No telemetry. The plugin runs no shell commands, starts no long-running processes (only
short-lived threads that send captures), and stores one file in your profile,
`conxt.json`. Conxt's [terms and privacy policy](https://conxt.dev/privacy) cover what it
stores.

## Tools only, without the plugin

Conxt also runs an MCP server at `https://mcp.conxt.dev/mcp/`. Hermes can connect to it with
an `mcp_servers` entry, which gives the agent Conxt's tools but no automatic recall or
capture: the model only uses MCP tools when it decides to. Use this plugin for memory that
works on every turn.

## Development

The tests run the provider against a stand-in Conxt server. They need Hermes on the path:

```bash
PYTHONPATH=/path/to/hermes-agent python -m pytest tests -q
hermes plugins validate . --install-deps
```

## License

MIT
