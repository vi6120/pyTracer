# Licensing

pyTracer is open source under the **Apache-2.0** license, every package. You can
embed it, run it, self-host it and build on it, including in closed-source and
commercial products. The full text is in [LICENSE](LICENSE); each package also
carries a copy.

| Package | License | What it is |
| --- | --- | --- |
| `pytracer-sdk` | Apache-2.0 | The instrumentation SDK you embed in your agent. |
| `pytracer-langgraph` | Apache-2.0 | Adapter for LangGraph / LangChain. |
| `pytracer-openai-agents` | Apache-2.0 | Adapter for the OpenAI Agents SDK. |
| `pytracer-crewai` | Apache-2.0 | Adapter for CrewAI. |
| `pytracer-agent-framework` | Apache-2.0 | Adapter for Microsoft Agent Framework. |
| `pytracer-claude-agent-sdk` | Apache-2.0 | Adapter for the Claude Agent SDK. |
| `pytracer-autogen` | Apache-2.0 | Adapter for AutoGen (archived upstream). |
| `pytracer-replay` | Apache-2.0 | The test and replay engine you run locally and in CI. |
| `pytracer-runner` | Apache-2.0 | The cross-branch runner you run locally and in CI. |
| `pytracer-cli` | Apache-2.0 | The command-line tool. |
| `pytracer-stores` | Apache-2.0 | Trace store (ClickHouse) and mock store (PostgreSQL) clients. |
| `pytracer-gateway` | Apache-2.0 | The ingestion server, self-hosted or hosted. |
| `pytracer-registry` | Apache-2.0 | The collaboration registry for shared test collections. |
| `pytracer-all` (meta) | Apache-2.0 | Installs every package above. |

## Earlier releases of the server packages

Up to and including 0.1.4, `pytracer-gateway` and `pytracer-registry` were
released under **AGPL-3.0-or-later**. Those published versions keep that license;
the change to Apache-2.0 applies from the next release onward. Every other
package has always been Apache-2.0.

## What is not open source

The hosted pyTracer service and its enterprise features (single sign-on, role-based
access, audit logs, retention controls, side-effect governance, the hosted registry
interface) are a separate, commercial product and are not in this repository.

## Contributions

Contributions are accepted under Apache-2.0. Contributors sign the
[Contributor License Agreement](CLA.md) once, on their first pull request.

## Trademarks

The license covers the code; the **pyTracer** name and logo are trademarks. See
[TRADEMARK.md](TRADEMARK.md).
