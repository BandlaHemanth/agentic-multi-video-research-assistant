# Deployment Guide: Agentic Multi-Video Research Assistant

This guide explains how to deploy the application to **Streamlit Community Cloud** and **Render**.

---

## ☁️ Streamlit Community Cloud

Streamlit Community Cloud is the fastest way to host Streamlit applications directly from a GitHub repository.

### Prerequisites
1. Push your code to a public GitHub repository. Ensure `.gitignore` ignores your local `.env` and `data/` directories.
2. Ensure `requirements.txt` contains all dependencies (including `faster-whisper` and `imageio-ffmpeg`).

### Step 1: Create a `packages.txt` File
Because the Whisper ASR fallback mechanism relies on `ffmpeg` to extract audio from YouTube videos, you must instruct Streamlit's container environment to install it.
Create a file named `packages.txt` at the root of your repository with the following contents:
```txt
ffmpeg
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [Streamlit Share](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **New app**.
3. Select your repository, branch (e.g., `main`), and the main file path: `app.py`.
4. Click the **Advanced settings** dropdown.
5. In the **Secrets** section, configure your environment variables:
   ```toml
   GOOGLE_API_KEY = "your_gemini_api_key_here"
   WHISPER_ASR_MODEL = "base"
   ```
6. Click **Deploy**. Your app will build and go live in a few minutes!

---

## 🚀 Render Deployment

Render is a robust cloud platform for hosting web applications. To guarantee that C++ libraries (for FAISS), PyTorch, and system utilities (`ffmpeg`) are bundled correctly without dependency conflicts, we recommend deploying using a **Docker service**.

### Step 1: Add a `Dockerfile` to your Project Root
Create a file named `Dockerfile` at the root of your project:

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8501

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Deploy on Render
1. Log in to [Render](https://render.com/).
2. Click **New** and select **Web Service**.
3. Connect your GitHub repository containing the application code.
4. Configure the Web Service settings:
   - **Name:** `agentic-video-research-assistant`
   - **Environment:** Select **Docker** (Render will automatically detect your `Dockerfile`).
   - **Instance Type:** Select a tier (the ASR fallback is CPU-heavy; select a tier with at least 2GB RAM. If deploying on a free instance, please configure `WHISPER_ASR_MODEL=tiny` to avoid OOM crashes).
5. Click **Advanced** to add Environment Variables:
   - `GOOGLE_API_KEY`: `your_gemini_api_key_here`
   - `WHISPER_ASR_MODEL`: `tiny` (or `base` if using a paid Render tier)
6. Click **Create Web Service**. Render will build the Docker container and deploy the app!
