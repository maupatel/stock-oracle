# Publishing Stock Oracle online — for free

The easiest free host for a Streamlit app is **Streamlit Community Cloud**. It runs
your app directly from a GitHub repo, gives you a public `https://…streamlit.app`
URL, and redeploys automatically every time you push. No servers, no credit card.

---

## Option A — Streamlit Community Cloud (recommended)

### 1. Put the project on GitHub (one time)
1. Create a free account at https://github.com
2. Install Git: https://git-scm.com/download/win
3. In this folder (`StockOracle`), open a terminal and run:
   ```powershell
   git init
   git add oracle_app.py requirements.txt .streamlit/config.toml oracle.ico
   git commit -m "Stock Oracle"
   ```
   > Don't commit the launcher files (`*.vbs`, `*.bat`, `*.lnk`) or `__pycache__` —
   > the included `.gitignore` already excludes them.
4. On GitHub click **New repository**, name it `stock-oracle`, **Public**, create it.
5. Copy the two "push an existing repository" commands GitHub shows, e.g.:
   ```powershell
   git remote add origin https://github.com/<your-username>/stock-oracle.git
   git branch -M main
   git push -u origin main
   ```

### 2. Deploy
1. Go to https://share.streamlit.io and sign in **with your GitHub account**.
2. Click **Create app → Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `<your-username>/stock-oracle`
   - **Branch:** `main`
   - **Main file path:** `oracle_app.py`
4. Click **Deploy**. First build takes ~3–5 min while it installs `requirements.txt`.
5. You get a public link like `https://stock-oracle.streamlit.app` — share it with anyone.

### 3. Updating later
Just push changes — the live app rebuilds automatically:
```powershell
git add -A
git commit -m "what changed"
git push
```

**Good to know**
- Free tier sleeps after inactivity; the first visit after that takes ~30s to wake.
- Reddit's API sometimes blocks cloud server IPs, so the **Social & Reddit** tab may be
  empty online even though it works locally — the app degrades gracefully (a notice, not a crash).
- yfinance, the yield curve, movers, and all six analysis modules work fine on the cloud.

---

## Option B — Hugging Face Spaces (also free)
1. Account at https://huggingface.co → **New Space** → SDK: **Streamlit**.
2. Upload `oracle_app.py`, `requirements.txt`, and the `.streamlit/` folder.
3. It builds and serves automatically at `https://huggingface.co/spaces/<you>/<name>`.

## Option C — Render.com (free web service)
Works but needs a start command: `streamlit run oracle_app.py --server.port $PORT --server.address 0.0.0.0`.
More setup than A or B; only worth it if you outgrow Community Cloud.

---

### Why not "just send someone the .exe"?
Streamlit is a web server, not a packaged desktop binary, so the clean way to share it
with other people is a hosted URL (Option A). The desktop shortcut in this folder is for
running it **on your own machine**.
