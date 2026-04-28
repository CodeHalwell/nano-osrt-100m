## 2025-04-28 - Tooltips and Autofocus

**Learning:** Gradio sliders benefit greatly from `info="..."` attributes to explain ML concepts to users, and text inputs should use `autofocus=True` to save a click for chat interfaces. Ensure all mock files created for frontend UI testing are cleaned up before submitting. Running `ruff format` on files we touch may bloat diffs with unrelated formatting changes, so be cautious to stick to the <50 line change limit.
**Action:** Use `info` for Gradio components with complex settings and `autofocus` on primary inputs. Clean up test files.
