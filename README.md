# Fish Habitat Zonation in the Tropical Indian Ocean via Hybrid PCA–Ensemble GMM Environmental Regime Classification

## Overview

This repository presents a data-driven framework for identifying potential fish habitat zones in the Tropical Indian Ocean through environmental regime classification. Instead of relying on fishery-dependent observations such as Catch Per Unit Effort (CPUE), the proposed framework utilizes multi-variable oceanographic conditions to identify ecologically meaningful marine regimes using an unsupervised learning approach.

The methodology integrates Principal Component Analysis (PCA) for dimensionality reduction with an Ensemble Gaussian Mixture Model (GMM) for robust environmental regime classification. The resulting ecological regimes are subsequently interpreted as potential fish habitat zones based on their physical and biological characteristics.

---

## Motivation

Traditional fish habitat studies frequently depend on fishery-dependent datasets such as CPUE, which are influenced by numerous external factors including fishing effort, fleet distribution, regulations, and weather conditions. Consequently, CPUE does not always accurately represent the true spatial distribution of marine habitats.

This project proposes a fishery-independent framework that identifies habitat zones solely from oceanographic environmental conditions, providing an objective and reproducible approach for marine habitat characterization.

---

## Study Area

- **Region:** Tropical Indian Ocean
- **Spatial Resolution:** 0.25°
- **Temporal Coverage:** Monthly observations (2015–2024)

Environmental variables used:

- Sea Surface Temperature (SST)
- Sea Surface Height (SSH)
- Salinity
- Dissolved Oxygen
- Chlorophyll-a Concentration

---

## Methodology

The overall workflow of the proposed framework is illustrated below.

```

Raw Environmental Data

↓

Preprocessing

↓

Standardization

↓

Principal Component Analysis (PCA)

↓

Bayesian Information Criterion (BIC)

↓

Ensemble Gaussian Mixture Model

↓

Consensus Environmental Regimes

↓

Fish Habitat Zonation

↓

Physical Oceanographic Interpretation

```

The proposed framework consists of the following major stages:

1. Data preprocessing and standardization
2. Principal Component Analysis (PCA)
3. Selection of the optimal number of clusters using BIC
4. Ensemble Gaussian Mixture Modeling
5. Consensus-based environmental regime generation
6. Habitat zone derivation
7. Physical interpretation of ecological regimes

---

## Repository Structure

```

Fish-Habitat-Zonation/

├── data/ # Dataset information
├── docs/ # Documentation and workflow diagrams
├── outputs/ # Figures and generated outputs
├── paper/ # Manuscript
├── src/ # Source code
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore

```

---

## Data Availability

The complete datasets used in this project are **not included** in this repository because they exceed GitHub's file size limitations (approximately 18 GB).

The datasets consist of monthly oceanographic products from Copernicus Marine Services covering 2015–2024.

After downloading the datasets, place them inside

```

data/raw/

```

and execute the preprocessing pipeline provided in the `src/preprocessing/` directory.

---

## Results

The proposed framework generates:

- Environmental regime maps
- Monthly fish habitat zonation maps
- Annual habitat distributions
- Physical characteristics of each ecological regime
- Habitat suitability interpretation

---

## Future Work

Future improvements include:

- Spatio-temporal ecological regime modeling
- Habitat uncertainty quantification
- Integration of fisheries-independent validation datasets
- Extension to species-specific habitat prediction

---

## Citation

If you use this repository in your research, please cite:

> *Citation will be updated upon publication.*

---

## License

This project is distributed under the MIT License.

---

## Contact

**Kamalesh Paramadayalan**

Vellore Institute of Technology (VIT), Chennai

Email: *(Add your institutional email after publication if desired.)*