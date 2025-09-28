# 🎙️ Caribbean Air Force – Daily News Agent

This project generates daily radio news scripts for **Caribbean Air Force Radio**.  
It fetches Caribbean news headlines, summarizes them with AI, and produces:

- **Bullet-point news briefs**
- **A timed 55-second radio script**
- **Social media captions**
- *(Optional)* An MP3 voiceover

Outputs are saved in the `/out` folder and can be viewed with the included Streamlit dashboard.

---

## 🚀 Quick Start

### 1. Clone or Download
Put all files (`agent.py`, `viewer.py`, `requirements.txt`) in a project folder.

### 2. Install Python Dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python agent.py --demo --log-level INFO
export OPENAI_API_KEY=sk-yourkeyhere   # Windows PowerShell: setx OPENAI_API_KEY "sk-..."
python agent.py --log-level INFO
streamlit run viewer.py
| Flag               | Description                                                |
| ------------------ | ---------------------------------------------------------- |
| `--demo`           | Use built-in sample headlines + script (offline safe)      |
| `--log-level`      | Control logging (DEBUG, INFO, WARNING)                     |
| `--init`           | Generate requirements.txt, .env.example, README_LAUNCH.txt |
| `--every 4h`       | Run every 4 hours continuously                             |
| `--schedule 07:30` | Run daily at 07:30 local time                              |
ngrok http 8501
/agent.py          # Main agent pipeline
/viewer.py         # Streamlit dashboard to browse output
/out/              # JSON + MP3 output files
/requirements.txt  # Python dependencies
README.md          # This file

---

### How to Create It
1. Open any text editor (VS Code, Notepad, etc.).
2. Paste the above markdown into a new file.
3. Save it as `README.md` in the root of your project.

If you put this on GitHub, the README will automatically appear as the project homepage.

---

Would you like me to generate this README.md for you and save it directly to your project folder (next to `agent.py` and `viewer.py`) so it’s ready to commit or share?
