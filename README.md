# Conxt memory for Hermes Agent

Governed long-term memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent),
from [Conxt](https://conxt.dev).

Your agent remembers across sessions, as most memory providers allow. Conxt also governs
what the agent remembers for a team:

- **Personal memory is yours.** What Hermes learns in your sessions is kept for you.
- **Team memory is approved by a person.** The agent can *propose* a memory to your team.
  Nobody recalls it until someone on the team approves it in Conxt. The plugin's API key
  cannot approve its own proposals.
- **Every read is on record.** Each recall is logged in Conxt with what was read and who
  read it.

## What it does

| When | What happens |
|---|---|
| Before each turn | Your message is sent to Conxt and the matching memories come back as context: your own, plus your team's approved ones if a team is set. Greetings and one-word replies are skipped. |
| After the conversation | Turns are buffered locally and sent to Conxt in batches: when the session ends, before Hermes compresses the context, when you switch session, at shutdown, and every 5 exchanges. Conxt then extracts memories from each batch. |
| When Hermes's built-in memory adds an entry | The same text is saved to your Conxt memory. |
| When the agent chooses | `conxt_search` looks something up. `conxt_remember` saves a personal memory. `conxt_propose` proposes a team memory, and only exists when a team is set. |

Scheduled (cron), subagent and flush runs are **never written to automatically**. If Conxt
is slow or unreachable, recall is skipped for that turn. Unsent turns wait for the next
attempt, up to 400 messages, after which the oldest are dropped. Nothing ever waits for a
person, so unattended runs cannot hang.

## Requirements

- Hermes Agent 0.21.6 or later.
- A Conxt API key. During the beta any free Conxt account can create one; after it, keys
  need a paid plan. Team memory needs the Team plan. Create the key from the **Hermes** card
  at [conxt.dev/connect-ide](https://conxt.dev/connect-ide): each tool gets its own key, so
  you can see what Hermes reads and revoke it without touching your other tools.

## Install and set up

The plugin is in the [Hermes plugin catalog](https://github.com/NousResearch/hermes-agent/tree/main/plugin-catalog):

```bash
hermes plugins install conxt
hermes memory setup        # choose "conxt", paste your API key, optionally a team id
```

To run the latest commit instead of the catalog's pinned version, install it by hand into
your profile:

```bash
git clone https://github.com/conxtdevelopmentteam/hermes-conxt "$HERMES_HOME/plugins/conxt"
hermes memory setup
```

Or set it in `config.yaml`:

```yaml
memory:
  provider: conxt
```

## Configuration

| Setting | Where | Default | Meaning |
|---|---|---|---|
| `CONXT_API_KEY` | secret (`.env`, via `hermes memory setup`) | — | Required. Read through Hermes's scoped secrets, per profile. |
| `CONXT_TEAM_ID` | env, or `team_id` in `$HERMES_HOME/conxt.json` | none | A Conxt team id. Adds the team's approved memories to recall and turns on `conxt_propose`. |
| `CONXT_BASE_URL` | env, or `base_url` in `$HERMES_HOME/conxt.json` | `https://mcp.conxt.dev` | The Conxt API. Must be `https`. |

Each Hermes profile has its own key, team and settings.

## What leaves your machine

Everything goes to the Conxt API at `https://mcp.conxt.dev` (or your `CONXT_BASE_URL`),
authenticated with your API key, and to nothing else.

- **Recall:** the text of your message (first 500 characters) and the team id, if set.
- **Capture:** the user and assistant messages of the conversation, up to 10,000 characters
  each. Tool calls and tool results are not sent, but anything the assistant repeats in its
  reply is: file paths, command output or code it quotes. Conxt redacts secrets it
  recognises before storing memories.
- **Saves and proposals:** the text of the memory, its type and a title.
- **Mirrored memory writes:** the text of each entry Hermes's built-in memory adds.

There is no telemetry. The plugin runs no shell commands, starts no long-running processes
(only short-lived threads that send captures), and stores nothing outside your profile.
Its one file there is `conxt.json`. Conxt's own terms and privacy policy cover what it
stores: [conxt.dev](https://conxt.dev).

## Tools only, without the plugin

Conxt also runs an MCP server, at `https://mcp.conxt.dev/mcp/`. Hermes can connect to it
directly with an `mcp_servers` entry, which gives the agent Conxt's tools. It does not give
automatic recall or capture: the model only uses MCP tools when it decides to. Use this
plugin for memory that works on every turn.

## Development

The tests run the provider against a stand-in Conxt server. They need Hermes on the path:

```bash
PYTHONPATH=/path/to/hermes-agent python -m pytest tests -q
hermes plugins validate . --install-deps
```

## License

MIT
