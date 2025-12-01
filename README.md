# Système de Pricing d'Options et Greeks

## Description

Système complet de valorisation d'options européennes utilisant le modèle Black-Scholes-Merton, avec calcul des Greeks et analyse de la volatilité implicite.

**Niveau** : M2 Finance Quantitative - Sorbonne Université

## Fonctionnalités

- **Black-Scholes-Merton** : Pricing Call/Put européennes
- **Greeks complets** : Delta, Gamma, Vega, Theta, Rho
- **Volatilité implicite** : Méthode Newton-Raphson
- **Volatility smile** : Analyse et visualisation
- **Hedging Delta-neutre** : Simulation de couverture dynamique
- **Validation** : Comparaison avec données marché CAC40

## Structure du Projet

```
options-greeks-system/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── black_scholes.py      # Modèle BSM et pricing
│   ├── greeks.py             # Calcul des Greeks
│   ├── implied_volatility.py # Volatilité implicite
│   ├── volatility_smile.py   # Analyse du smile
│   └── delta_hedging.py      # Couverture Delta-neutre
├── tests/
│   └── test_pricing.py       # Tests unitaires
└── examples/
    └── demo.py               # Démonstration avec données CAC40
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation Rapide

```python
from src.black_scholes import black_scholes_price
from src.greeks import calculate_all_greeks

# Prix d'un Call européen
S = 100      # Prix spot
K = 100      # Strike
T = 1.0      # Maturité (1 an)
r = 0.05     # Taux sans risque
sigma = 0.2  # Volatilité

prix_call = black_scholes_price(S, K, T, r, sigma, option_type='call')
print(f"Prix du Call : {prix_call:.4f} €")

# Greeks
greeks = calculate_all_greeks(S, K, T, r, sigma, option_type='call')
print(f"Delta : {greeks['delta']:.4f}")
print(f"Gamma : {greeks['gamma']:.4f}")
print(f"Vega  : {greeks['vega']:.4f}")
```

## Références Théoriques

- Hull, J.C. (2018). *Options, Futures, and Other Derivatives*, 9th Edition
  - Chapitre 15 : Modèle Black-Scholes-Merton
  - Chapitre 19 : Greeks
  - Chapitre 20 : Volatilité implicite et smile

## Résultats

- Écart < 0.5% entre prix calculés et prix de marché
- Testé sur 200+ options CAC40

## Auteur

Projet M2 Finance Quantitative - Sorbonne Université
