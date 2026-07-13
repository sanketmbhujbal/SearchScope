# SearchScope demo app

An interactive presentation of this project's real findings: the
pipeline metrics, the label-leak before/after, SHAP vs. ablation, a
real mined vocabulary-mismatch case, the personalization results by
role, and the 7 real QA rejection-gate cases.

This app does not run the live retrieval/reranking/LTR pipeline. That
stack needs Pyserini (a JVM-backed BM25 index), FAISS, PyTorch, and
XGBoost, the same heavy, environment-fragile dependencies that caused
most of the real engineering friction earlier in this project (see the
main README's Day 1-2 setup notes). Reintroducing that stack into a
free-tier hosted demo would trade a small amount of interactivity for a
much larger risk of the demo itself breaking. Instead, every number and
example here is real, pulled directly from the project's own
`results/*.md` files, and `data.py` cites exactly which file each one
came from.

## Run locally

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Deploy for free (Streamlit Community Cloud)

1. Push this repo to a public GitHub repository (the whole repo, not
   just this folder, since `data.py` and `app.py` are self-contained
   but the deploy step needs a repo to point at).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
   with GitHub.
3. Click "New app," pick this repository, and set:
   - **Main file path:** `streamlit_app/app.py`
   - Everything else (Python version, requirements) is auto-detected
     from `streamlit_app/requirements.txt`.
4. Deploy. You'll get a permanent `https://<something>.streamlit.app`
   URL you can share directly, no server, no Docker, no domain setup.

## Updating the data

Every number in `data.py` is hand-copied from a `results/*.md` file with
a comment noting the source. If a findings file changes (a re-run
produces a different number, a new case is mined), update `data.py` to
match, rather than letting the two drift apart. There's no automated
sync between them by design, since the demo app is deliberately decoupled
from the heavy pipeline that produces those files in the first place.
