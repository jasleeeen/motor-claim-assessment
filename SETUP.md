# Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m pip install -e ".[vision]"
```

Open `.env` and paste your Groq key after `GROQ_API_KEY=`.

The detector weights are too large for GitHub. Download both files from
[this Drive folder](https://drive.google.com/drive/folders/1tV1ffWqlYVU-x_ZSpFASSvHl9e22QlkP?usp=sharing)
and copy them into the `weights` folder.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Some tests are skipped because the raw policies are not in the project folder. That is
expected, and skips are not failures.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Upload a PDF from `data\sample_policies\`. The test images are in `data\test_images\`.
Fill in a claim and submit.
