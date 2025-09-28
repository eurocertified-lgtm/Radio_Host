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
