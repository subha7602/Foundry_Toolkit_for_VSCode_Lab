# 🚀 AI Agents Hands-On Workshop: Attendee Pre-Setup Guide

> **⏱️ Estimated Setup Time:** ~10 minutes  
> **📅 Please complete this guide 1 day before the workshop** so you are 100% ready to build agents on day one without Wi-Fi download delays!

---

## 📋 Checklist Overview
- [ ] 1. Install Core Tools (Python 3.10+, Git, VS Code)
- [ ] 2. Clone the Workshop Repository
- [ ] 3. Choose your LLM Engine (**Option A: Groq Cloud** or **Option B: Local Ollama**)
- [ ] 4. Setup Python Environment & Dependencies
- [ ] 5. Run the 30-Second Verification Test

---

## 🛠️ Step 1: Core Tools Verification

Make sure you have the following installed on your laptop:

1. **Python 3.10, 3.11, or 3.12**
   - Verify in terminal:
     ```bash
     python3 --version   # (On Windows: python --version)
     ```
   - *If not installed:* Download from [python.org/downloads](https://www.python.org/downloads/).
2. **Git**
   - Verify: `git --version`
   - *If not installed:* Download from [git-scm.com](https://git-scm.com/).
3. **VS Code** (or your favorite code editor)
   - Download from [code.visualstudio.com](https://code.visualstudio.com/).

---

## 📦 Step 2: Clone the Repository

Open your terminal or command prompt and clone the workshop repository:

```bash
git clone https://github.com/microsoft-foundry/Foundry_Toolkit_for_VSCode_Lab.git
cd Foundry_Toolkit_for_VSCode_Lab
```

---

## 🔑 Step 3: Choose Your AI Engine (Pick Option A or B)

You can choose whichever option fits your workflow best:

---

### 🅰️ Option A: Groq Cloud (Recommended for Speed & Live Demos)
*Ultra-fast inference (<1 second per agent turn). 100% free with no credit card required.*

1. Go to **[https://console.groq.com/keys](https://console.groq.com/keys)** and sign up with GitHub / Google.
2. Click **Create API Key**, give it a name, and copy your key (`gsk_...`).
3. Save this key for Step 4.

---

### 🅱️ Option B: Local Ollama (Recommended for 100% Offline / Zero Cloud)
*Runs models locally on your laptop's CPU/GPU. No internet required during the workshop.*

1. **Install Ollama:**
   - **Mac:** `brew install ollama` (or download from [ollama.com/download](https://ollama.com/download))
   - **Windows:** Download the Windows installer from [ollama.com/download/windows](https://ollama.com/download/windows)
   - **Linux:** `curl -fsSL https://ollama.ai/install.sh | sh`
2. **Start the Ollama app / daemon:**
   ```bash
   ollama serve
   ```
3. **Pre-download the workshop model (Please do this at home before arriving!):**
   ```bash
   # For standard laptops (16GB RAM) - Best tool calling accuracy:
   ollama pull qwen2.5:7b

   # OR for lightweight laptops (8GB RAM / non-GPU):
   ollama pull llama3.2:3b
   ```

---

## ⚙️ Step 4: Environment & Dependency Setup

### 1. Configure Lab 01 (Single Agent)
```bash
cd workshop/lab01-single-agent/agent

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Create your .env file
cp .env.example .env
```

Open `.env` in your editor and paste your credentials:
- If using **Groq (Option A):** Paste your `GROQ_API_KEY="gsk_..."`
- If using **Ollama (Option B):** Ensure `OLLAMA_BASE_URL="http://localhost:11434/v1"`

---

### 2. Configure Lab 02 (Multi-Agent Workflow)
```bash
cd ../../lab02-multi-agent/PersonalCareerCopilot

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Create your .env file
cp .env.example .env
```
Paste your same `GROQ_API_KEY` (or Ollama settings) in this `.env` file as well.

---

## 🧪 Step 5: Smoke Test (Verify Everything Works!)

### Test 1: Single Agent (Lab 01)
```bash
cd workshop/lab01-single-agent/agent
source .venv/bin/activate

# If using Groq:
python3 groq_main.py

# If using Ollama:
python3 ollama_main.py
```
✅ **Expected:** You will see the agent format a sample technical incident into a 3-part Executive Summary with today's date stamped via a tool call!

---

### Test 2: Multi-Agent Workflow (Lab 02)
```bash
cd workshop/lab02-multi-agent/PersonalCareerCopilot
source .venv/bin/activate

# If using Groq:
python3 groq_main.py

# If using Ollama:
python3 ollama_main.py
```
✅ **Expected:** You will see all 4 agents (`ResumeParser` ➔ `JobDescriptionAgent` ➔ `MatchingAgent` ➔ `GapAnalyzer`) execute sequentially and output an upskilling learning roadmap!

---

## ❓ Frequently Asked Questions & Troubleshooting

* **Q: `zsh: command not found: pip` or `python`?**  
  *Fix:* On macOS and Linux, always use `python3` and `pip3`, or make sure your virtual environment is active (`source .venv/bin/activate`).

* **Q: `Ollama connection error`?**  
  *Fix:* Make sure `ollama serve` is running in another terminal window or the Ollama desktop app icon is in your taskbar.

* **Q: Do I need an Azure account or credit card?**  
  *Fix:* **No.** Groq has a free tier without a credit card, and Ollama runs 100% locally on your machine.

---

### 🎉 You are now ready for the workshop! See you there!
