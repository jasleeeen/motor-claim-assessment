# Setup (Windows PowerShell in Integrated Terminal)

py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m pip install -e ".[vision]"

# Open .env and paste your Groq key 
# Drive link - https://drive.google.com/drive/folders/1tV1ffWqlYVU-x_ZSpFASSvHl9e22QlkP?usp=sharing
# From the drive link, copy the files into the weights folder, they are too big to be uploaded to github directly

.\.venv\Scripts\python.exe -m pytest -q
# some tests will be skipped as raw policies are not added to the project folder

.\.venv\Scripts\python.exe -m streamlit run app.py

# Upload a PDF from data\sample_policies\, and the test images are in data\test_images\
# fill in a claim, submit. 
