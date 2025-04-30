# 🧭 PTSP-MOGA-NSGA-II

**Period Travelling Salesman Problem (PTSP) solved with NSGA-II**

This repository contains an implementation of a Multi-Objective Genetic Algorithm (NSGA-II) to solve the PTSP over a 7-day planning horizon. The goal is to generate weekly routing plans that:

- 🛣️ **Minimize total travel distance**  
- ⚖️ **Minimize demand imbalance across days**

---

## 🔧 Key Features

- 📍 **Problem Formulation**  
  • City coordinates (x, y)  
  • Visit frequency requirements  
  • Day-wise demand profiles  

- 🎯 **Bi-Objective Optimization**  
  • Total distance minimization  
  • Demand-imbalance minimization  

- 🧬 **NSGA-II Implementation**  
  • Custom chromosome encoding  
  • PMX crossover & shuffle mutation  
  • Fast non-dominated sorting & crowding-distance  

### 📊 Visualization of Metrics Over Generations

**Hypervolume (HV)**  
<img src="https://github.com/user-attachments/assets/6a4f33e5-87ae-4958-b80b-8d984548ed14" width="600"/>

**Spread (Δ) & Spacing (Sp)**  
<img src="https://github.com/user-attachments/assets/d9656a36-17d3-469e-9b58-ee0ed9982210" width="600"/>

**Diversity**  
<img src="https://github.com/user-attachments/assets/043b6767-101d-40d3-88c7-03cd1277d397" width="600"/>

**Pareto Front Size (Initial vs Final)**  
<img src="https://github.com/user-attachments/assets/9872f4b4-1d5f-4c72-a146-abe9257bd98b" width="600"/>

---

### 🗺️ VRP Dataset Map Overview

**VRP-8 Cities Layout**  
<img src="https://github.com/user-attachments/assets/846b4aa0-8578-4e36-9356-6aba586866ce" width="600"/>


---
```text
PTSP-MOGA-NSGA-II/
├── vrp8/                      # VRP-8 dataset (coordinates & demands)
├── results/                   # Saved populations, logbooks, and plots
├── main.py                    # Entry point for train/eval
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview
```
---
## 📈 Results

### Salesman Routes Before vs After Optimization

<table>
  <tr>
    <td align="center">
      Before Optimization<br>
      <img src="https://github.com/user-attachments/assets/c8cbbdf5-db3e-4e8b-9375-5d8687c36d63" width="350"/>
    </td>
    <td align="center">
      After Optimization<br>
      <img src="https://github.com/user-attachments/assets/fa3d11a6-4d5f-41ea-a2ba-4c2a0523a3fb" width="350"/>
    </td>
  </tr>
</table>

---

### Representative Pareto Solutions

<table>
  <tr>
    <td align="center">
      Min Total Distance<br>
      <img src="https://github.com/user-attachments/assets/42f6d186-6bcc-45ff-b3e9-42a0507d31a3" width="300"/>
    </td>
    <td align="center">
      Min Demand Imbalance<br>
      <img src="https://github.com/user-attachments/assets/f08023c8-ba74-4e37-b176-c84179182e16" width="300"/>
    </td>
    <td align="center">
      Best Balanced<br>
      <img src="https://github.com/user-attachments/assets/5e24a08a-9aa8-4f76-9e94-5f91f6214ff8" width="300"/>
    </td>
  </tr>
</table>

