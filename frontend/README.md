# PoliEmotiFusion — Multimodal Political Emotion Intelligence Frontend

A deep learning multimodal emotion analysis platform designed for political discourse. Analyze emotional signals across **Text**, **Image**, **Audio**, and **Video** using state-of-the-art model architectures, timeline tracking, and interactive visualizations.

---

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Quick Start Guide](#-quick-start-guide)
- [Available Scripts](#-available-scripts)
- [Application Routes](#-application-routes)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)

---

## 🌟 Features

- **Text Emotion Analysis (`/text`)**: Evaluates political statements, speeches, and social commentary using transformer architectures (BERT / RoBERTa).
- **Image Scene Emotion (`/image`)**: Uses CLIP zero-shot similarity to score the overall emotion conveyed by a political image; scores are experimental and not calibrated accuracy.
- **Audio / Speech Emotion (`/audio`)**: Analyzes acoustic features, pitch, and vocal cadence using Wav2Vec2 / Audio Spectrogram Transformers.
- **Video Temporal Emotion (`/video`)**: Real-time temporal emotion tracking over video timelines using 3D CNNs and SlowFast networks.
- **Interactive Dashboard (`/`)**: Comprehensive overview of emotion telemetry, activity feeds, and model latency metrics.
- **Insights & Metrics (`/insights`)**: Cross-modality emotion distribution radar charts, confidence matrices, and aggregated political discourse trends.
- **Analysis History (`/history`)**: Searchable and filterable history of past multimodal analyses with JSON/CSV export capability.
- **Architecture Explorer (`/architecture`)**: Visual documentation of the late-fusion and multi-tower deep learning pipeline.

---

## 🛠️ Tech Stack

- **Framework**: [TanStack Start](https://tanstack.com/start) (Full-stack SSR / Vite / Nitro engine)
- **Routing**: [TanStack Router](https://tanstack.com/router) (Type-safe file-based routing)
- **State & Caching**: [TanStack Query](https://tanstack.com/query)
- **UI & Styling**: [React 19](https://react.dev/), [Tailwind CSS v4](https://tailwindcss.com/), [Radix UI](https://www.radix-ui.com/)
- **Charts & Visualizations**: [Recharts](https://recharts.org/)
- **Icons**: [Lucide React](https://lucide.dev/)
- **Language**: TypeScript 5.8+

---

## ⚙️ Prerequisites

Ensure you have the following installed on your machine:

- **Node.js**: `v18.0.0` or higher (Recommended: `v20.x` or `v22.x`)
- **npm**: `v9.0.0` or higher (Bundled with Node.js)
  - *Alternatively, you can use **Bun** or **pnpm** if installed.*

To verify your Node.js and npm versions:

```bash
node -v
npm -v
```

---

## 🚀 Quick Start Guide

Follow these steps to get the frontend running locally:

### 1. Open Terminal and Navigate to the Frontend Directory

If you are in the project root directory:

```bash
cd frontend
```

### 2. Install Dependencies

Install all required packages from `package.json`:

```bash
npm install
```

> **Note**: If you have [Bun](https://bun.sh) installed, you can optionally run:
> ```bash
> bun install
> ```

### 3. Start the Local Development Server

Run the development server with hot-module reloading:

```bash
npm run dev
```

### 4. Open in Your Browser

Once Vite starts, open your browser and navigate to:

```text
http://localhost:5173
```
*(If port 5173 is occupied, Vite will automatically select the next available port, e.g., 5174 or 3000).*

---

## 📜 Available Scripts

In the `frontend` directory, you can run:

| Command | Description |
| :--- | :--- |
| `npm run dev` | Starts the local Vite development server with HMR. |
| `npm run build` | Compiles and builds the production-ready SSR & static bundles via Vite & Nitro. |
| `npm run build:dev` | Builds the project in development mode for debugging. |
| `npm run preview` | Locally previews the production build. |
| `npm run lint` | Runs ESLint to inspect code quality and syntax errors. |
| `npm run format` | Formats all code files using Prettier. |

---

## 🗺️ Application Routes

| Route | Page Name | Description |
| :--- | :--- | :--- |
| `/` | **Dashboard** | Overview of modalities, quick stats, and recent emotion trends |
| `/text` | **Text Analyzer** | Input political text or speech transcripts for emotion breakdown |
| `/image` | **Image Analyzer** | Upload political images for whole-scene emotion scoring |
| `/audio` | **Audio Analyzer** | Upload speech audio files with waveform visualization |
| `/video` | **Video Analyzer** | Upload video clips for frame-by-frame temporal emotion tracking |
| `/insights` | **Insights & Trends** | Aggregate charts, emotion radars, and model performance metrics |
| `/history` | **History** | Log of previous inference queries with search and filters |
| `/architecture`| **Architecture** | System pipeline, feature extractors, and fusion mechanics |
| `/about` | **About** | Project background, dataset references, and ethics |
| `/settings` | **Settings** | UI preferences and local workspace options |

---

## 📂 Project Structure

```text
frontend/
├── public/                 # Static public assets
├── src/
│   ├── components/         # Reusable UI components
│   │   ├── common/         # Shared page and statistic components
│   │   ├── layout/         # AppShell, Navigation sidebar, headers
│   │   └── ui/             # Radix UI primitives (buttons, dialogs, cards)
│   ├── hooks/              # Custom React hooks
│   ├── lib/                # Emotion color mappings, utilities, types
│   ├── routes/             # TanStack Router route definitions
│   │   ├── __root.tsx      # Root application layout & shell
│   │   ├── index.tsx       # Main dashboard
│   │   ├── text.tsx        # Text modality page
│   │   ├── image.tsx       # Image modality page
│   │   ├── audio.tsx       # Audio modality page
│   │   ├── video.tsx       # Video modality page
│   │   ├── insights.tsx    # Emotion trends & charts
│   │   └── ...
│   ├── services/           # Backend API calls and application configuration
│   ├── server.ts           # SSR entry & server handlers
│   ├── start.ts            # TanStack Start configuration
│   └── styles.css          # Global Tailwind CSS imports & theme tokens
├── package.json            # Dependencies and npm scripts
├── tsconfig.json           # TypeScript configuration
└── vite.config.ts          # Vite & TanStack configuration
```

---

## 🔧 Troubleshooting

### 1. `'vite' is not recognized as an internal or external command`
This indicates that dependencies have not been installed yet.
**Fix**:
```bash
npm install
```

### 2. Port Already in Use
If port `5173` is busy, specify an alternative port:
```bash
npx vite --port 3000
```

### 3. Clear Cache / Fresh Install
If you run into dependency conflicts or stale cache:
```bash
# Windows PowerShell
Remove-Item -Recurse -Force node_modules, package-lock.json
npm install
```
