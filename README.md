# 👗 Multimodal Fashion & Context Retrieval
### Glance ML Internship Assignment

This project implements an intelligent **multimodal image retrieval system** that retrieves fashion images based on **natural language queries**. The system understands not only *what* a person is wearing, but also *where* they are and the *style/vibe* of the outfit.

The solution focuses on **ML logic, compositional reasoning, and fashion-specific retrieval**, going beyond a vanilla CLIP-based approach.

---

## 📌 Problem Statement

Given a natural language query such as:
- *"A red tie and a white shirt in a formal setting"*
- *"Casual weekend outfit for a city walk"*

the system retrieves the **top-k most relevant images** from a diverse fashion dataset, accounting for:
- Clothing attributes
- Color composition
- Context / environment
- Style and intent

---

## 📂 Dataset

A dataset of **500–1,000 fashion images** was sourced/simulated with variations across three axes:

### 1️⃣ Environment
- Office interiors
- Urban streets
- Parks
- Home settings

### 2️⃣ Clothing Types
- Formal (blazers, button-downs, ties)
- Casual (t-shirts, hoodies)
- Outerwear (jackets, raincoats)

### 3️⃣ Color Palette
- Wide range of garment colors
- Multi-color compositions (e.g., shirt + pants + accessories)

Datasets such as **Fashionpedia** were used as reference for annotation and attribute diversity.

---

## 🧠 System Architecture

The solution is divided into **two clear ML workflows**, implemented modularly in a single repository.

