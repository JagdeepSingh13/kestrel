# Glossary

Words used the same way in the README, the docs, the code and the shell.

## An agent
- **Model:** the language model that reasons and decides. Kestrel never runs it; a backend does.
- **Backend:** the program that runs the model and answers HTTP requests (Ollama, LM Studio, llama.cpp, mlx-lm, vLLM).
- **Harness:** everything around the model. Kestrel is the harness.
- **Context:** what the agent knows: its agent file and the files in the workspace. Also, per call, everything sent to the model.
- **Agent:** a YAML file naming the model, instructions, tools, permissions and limits.
- **Workspace:** the one folder agents may touch.

## Parts of the harness
- **Context (building it):** fitting instructions, task, conversation and tool results into the model's window.
- **Control:** budgets and stop reasons; when a run ends.
- **Safety gate:** the checks before a tool runs: permission (allow, confirm, deny), path jail, argument check.
- **Action:** running a tool.
- **Results back:** tool output returned to the model, marked untrusted.
- **Run log:** the file every step is written to.
- **Inspect and replay:** reading a run with `/trace`, or feeding it back through the loop without a model.

## What gets recorded
- **Step:** one thing that happened in a run (task received, model call, tool call and result, final answer). Stored as events.
- **Run:** one task from start to finish. One log file.
- **Session:** one shell conversation and its runs (planned).

## Getting better
- **Repair:** turning a small model's broken tool call into a valid one (planned).
- **Model ladder / step up:** handing one stuck step to a bigger model, then stepping back down (planned).
- **Test (eval):** a task plus a check on the answer, the workspace files, or the tools called (planned).
- **Recipe:** a successful run replayed without the model except at real decisions (planned).
- **Autotune:** trying harness settings on your tests and proposing the best for your model (planned).
