# Customer Channel Affinity & Next-Best-Channel Prediction  
*(Rules-Based + Machine Learning)*

## 📌 Project Overview
This project builds an **end-to-end customer channel affinity system** to identify each customer’s **preferred marketing channel** and **predict their next-best channel** using both:
- a **rules-based scoring engine** (descriptive, explainable), and  
- a **machine-learning model** (predictive, pattern-driven).

The system enables **comparison between business-rule logic and ML-driven predictions**, supporting more effective personalization, media planning, and campaign targeting.

---

## 🎯 Business Objective
Marketing teams often struggle to determine:
- which channel a customer truly prefers,
- which channel they are most likely to engage with next,
- and whether rules-based heuristics or ML models perform better.

This project answers:
- **“What is this customer’s strongest channel affinity?”**
- **“What is the best channel to target this customer in the next 60 days?”**
- **“How do rules-based and ML approaches differ?”**

---

## 📊 Data Overview
The solution uses **8+ real-world marketing datasets**, covering the full customer journey:

| Dataset | Description |
|------|------------|
| `customers` | Customer demographics, acquisition channel, LTV, purchase history |
| `sessions` | Web/app sessions with engagement metrics |
| `marketing_touchpoints` | Impressions, clicks, attributed revenue by channel |
| `email_engagement` | Sends, opens, clicks, time-to-open |
| `content_engagement` | Video watch time, completion, engagement events |
| `social_media_engagement` | Platform-level engagement and sentiment |
| `orders` | Transaction-level revenue |
| `order_items` | Product-level order details |
| `campaigns_meta` | Campaign metadata (channel, budget, targeting) |

**Scale**
- ~40,000 customers  
- ~200,000 sessions  
- Multi-channel coverage: Organic Search, Google Ads, YouTube, Facebook, Instagram, Email, Direct

---

## 🧠 Approach Overview

### 1️⃣ Rules-Based Channel Affinity Engine
A **transparent, explainable scoring system** built using business logic.

**Key characteristics**
- Customer-level feature aggregation using pandas
- Channel-specific weighted scoring
- Min-max normalization for comparability
- Fully interpretable outputs

**Features used**
- Email: open rate, click rate, time-to-open  
- Sessions: engagement score, session count  
- Content: video watch time, completion rate  
- Social: platform-level engagement & sentiment  
- Revenue: channel-attributed revenue  
- Recency: inverse recency weighting  

**Output**
- Normalized affinity scores per channel  
- `preferred_channel` per customer  

---

### 2️⃣ ML-Based Channel Affinity Model
A **multiclass predictive model** that learns behavioral patterns directly from data.

**Label design**
- Target channel derived from a **60-day future window**
- Label score combines:
  - touchpoint frequency  
  - attributed revenue  
  - engagement strength  

**Feature engineering**
- 90-day rolling behavioral window
- 100+ temporal features including:
  - channel-level session behavior  
  - marketing exposure  
  - engagement velocity  
  - recency signals  
  - last-touch & short-term channel dominance (7d / 30d)

**Model**
- Multiclass **LightGBM**
- One-hot encoding for categorical features
- Standard scaling for numeric features

**Performance**
- **Accuracy:** 82.6%  
- **Macro F1:** 0.82  

**Outputs**
- Predicted next-best channel per customer  
- Per-channel probability scores  

---

## 🔁 Rules-Based vs ML-Driven
| Aspect | Rules-Based | ML-Based |
|----|----|----|
| Explainability | High | Medium |
| Pattern discovery | Manual | Learned |
| Adaptability | Low | High |
| Business trust | Strong | Growing |
| Predictive power | Moderate | Strong |

This project intentionally supports **side-by-side comparison** to help teams decide **when to trust rules vs ML**.

---

## 🛠️ Tools & Technologies
- **Python** (pandas, numpy, scikit-learn)
- **LightGBM**
- **Excel** (exploratory analysis & validation)
- Feature engineering with rolling time windows
- End-to-end pipeline design (training → inference)

---

## 📁 Project Structure
```text
├── data/
│   └── comprehensive_data/
│       ├── customers.csv
│       ├── sessions.csv
│       ├── marketing_touchpoints.csv
│       ├── email_engagement.csv
│       ├── content_engagement.csv
│       ├── social_media_engagement.csv
│       ├── orders.csv
│       └── order_items.csv
│
├── output/
│   ├── channel_affinity_results.csv        # rules-based output
│   └── channel_affinity_predictions.csv    # ML predictions
│
├── rules_based_channel_affinity.py
├── ml_channel_affinity.py
└── README.md
