# 💻 Ollama Pre-Requisite Setup Guide (Pre-Workshop)

---


## 📋 Minimum Requirements
- **OS:** macOS (Apple Silicon or Intel), Windows 10/11, or Linux
- **RAM:** 8 GB minimum (16 GB recommended)
- **Disk Space:** ~6–8 GB free storage

---

## 🛠️ Step 1: Install Ollama

### 🍎 macOS
- **Using Homebrew (Recommended):**
  ```bash
  brew install ollama
  ```
- **Or Direct Download:** Download and install the Mac app from [ollama.com/download](https://ollama.com/download).

### 🪟 Windows
- Download and run the official Windows installer from [ollama.com/download/windows](https://ollama.com/download/windows).

### 🐧 Linux
- Run the one-line terminal installer:
  ```bash
  curl -fsSL https://ollama.ai/install.sh | sh
  ```

---

## 🚀 Step 2: Start the Ollama Service

Make sure the Ollama background engine is running:

- **Mac / Windows (Desktop App):** Launch the **Ollama** application (you'll see an Ollama icon in your menu bar / taskbar).
- **Or via Terminal (Mac / Linux):**
  ```bash
  ollama serve
  ```

---

## 📥 Step 3: Pre-Download Workshop Models (Important)

Open your terminal or command prompt and pull the model weights:

### 1️⃣ Primary Model (Recommended for 16GB RAM):
```bash
ollama pull qwen2.5:7b
```

### 2️⃣ Lightweight Fallback Model (For 8GB RAM / Older CPUs):
```bash
ollama pull llama3.2:3b
```

---

## ✅ Step 4: Quick 5-Second Verification Test

Run a quick test prompt directly in your terminal to verify that your local model works:

```bash
ollama run qwen2.5:7b "Hello! Confirm you are running locally."
```

If it responds, type `/bye` or press `Ctrl+D` to exit.

---

### 🎉 Setup Complete!
You now have your local AI runtime and models ready for the workshop.
