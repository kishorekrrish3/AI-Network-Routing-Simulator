---

# 🤖 AI Network Routing Simulator

A Streamlit-based interactive simulator comparing **Traditional Dijkstra Routing** with **AI-based Q-Learning Routing** in a dynamic network environment with congestion handling, real-time visualization, and adaptive learning behavior.

---

## 📌 Table of Contents

* [About the Project](#about-the-project)
* [Features](#features)
* [How It Works](#how-it-works)
* [Project Structure](#project-structure)
* [Installation & Setup](#installation--setup)
* [Usage Guide](#usage-guide)
* [Technical Details](#technical-details)
* [Educational Value](#educational-value)
* [Author](#author)

---

## 📖 About the Project

This project demonstrates the difference between **Traditional Routing Algorithms** (Dijkstra) and **AI-based Adaptive Routing** (Q-Learning). Users can simulate network congestion, train an AI router, visualize routing paths, and compare performance side-by-side.

The system uses:

* **NetworkX** for network graphs
* **Matplotlib** for plotting
* **Streamlit** for UI
* **Q-Learning** for adaptive routing
* **Custom simulation logic** for congestion-driven path costs

A perfect tool for networking demonstrations, AI coursework, and real-time routing visualization.

---

## ✨ Features

### 🖥️ Simulator Features

* Visual interactive **network topology** (6-node graph)
* Adjustable **congestion scenarios**
* Traditional routing (Dijkstra)
* AI routing (Q-Learning):

  * Exploration vs exploitation
  * Reward-based learning
  * Adaptive rerouting under congestion

### 📊 Comparative Analysis

* Cost-based comparison
* Path visualization
* Auto-generated comparison tables
* Live training progress bar

### 🎮 User Controls

* Select source and destination
* Add/remove congestion
* Apply predefined scenarios
* Reset network & AI learning
* One-click “Compare Both Methods”

---

## ⚙️ How It Works

### 🔄 Traditional Routing (Dijkstra)

* Selects the shortest path using **base link costs only**
* Ignores congestion
* Produces the same path every time

### 🧠 AI Routing (Q-Learning)

* Learns routing behavior through repeated episodes
* Rewards:

  * Negative cost for long or congested paths
  * Bonus for reaching destination
* Chooses routes based on learned Q-values
* Adapts when congestion appears

---

## 📁 Project Structure

```
ai-routing-simulator/
│
├── app.py             # Main Streamlit application (full simulation code)
└── requirements.txt   # Python dependencies
```

> Only two files — simple, clean, and easy to deploy.

---

## 🚀 Installation & Setup

### Prerequisites

* Python 3.8+
* pip

### 1️⃣ Install packages

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the simulator

```bash
streamlit run app.py
```

### 3️⃣ Open in browser

```
http://localhost:8501
```

---

## 🧭 Usage Guide

### 1. Select Nodes

Choose a **source** and **destination** from the sidebar.

### 2. Add Congestion

Use:

* Predefined scenarios
* Manual link congestion controls

### 3. Traditional Routing

Click **Find Traditional Path** to see:

* Route
* Total cost
* Visualization

### 4. AI Routing

* Adjust training episodes
* Click **Train AI & Find Path**
* View:

  * Adaptive path
  * Learned cost
  * Training progress

### 5. Compare Both Methods

See a table with:

* Paths
* Costs
* Adaptiveness
* Winner

---

## 🔧 Technical Details

### Network

* 6-node topology (A–F)
* Weighted edges
* Congestion triples the cost

### Algorithms

* **Dijkstra** for fixed routing
* **Q-Learning** with:

  * ε-greedy selection
  * Reward shaping
  * Discount factor (γ = 0.9)
  * Learning rate (α = 0.1)

### Visualization

* NetworkX graph
* Congested edges = red
* Chosen path = blue

---

## 🎓 Educational Value

Great for:

* AI/ML fundamentals
* Reinforcement Learning demonstrations
* Networking & routing courses
* Comparing adaptive vs deterministic systems
* Live classroom demos

---

## 👤 Author

**Kishore P**
AI & Full Stack Developer
VIT Chennai

---



