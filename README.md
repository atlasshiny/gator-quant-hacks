# Gator Quant Hacks Project
A project competing in the 2026 Gator Quant Hacks hackathon under the **Systematic Trading** path.

# Get Started

Follow these steps to set up your development environment, configure API keys, and run the pipeline locally or on HiPerGator.

## Prerequisites

- **Python**: Version `3.11` or higher.
- **Package Manager**: [`uv`](https://github.com/astral-sh/uv) (recommended for ultra-fast dependency management).
- **HiPerGator Access**: Valid UF Research Computing account with Slurm submission privileges (if running HPC jobs).

## 1. Environment Setup

### Local Setup (Using `uv`)

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-org/gator-quant-hacks.git](https://github.com/your-org/gator-quant-hacks.git)
   cd gator-quant-hacks
   ```

2. **Create and sync the virtual environment:**
   ```bash
   uv sync
   ```
   *This creates a `.venv` directory and installs all locked dependencies using `uv.lock`.*

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

## 2. API Credentials Configuration

Create a `.env` file in the root directory of the project:

```bash
cp .env.example .env
```

Open `.env` and fill in your API credentials:

```env
# Databento
DATABENTO_API_KEY=db-your_databento_api_key_here
DATABENTO_BASE_URL=[https://hist.databento.com](https://hist.databento.com)
DATABENTO_DEFAULT_DATASET=GLBX.MDP3

# FRED
FRED_API_KEY=your_fred_api_key_here
FRED_BASE_URL=[https://api.stlouisfed.org/fred](https://api.stlouisfed.org/fred)
```
