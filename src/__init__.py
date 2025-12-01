# -*- coding: utf-8 -*-
"""
Options Greeks System
=====================
Système de pricing d'options et calcul des Greeks.

Modules:
    - black_scholes: Modèle Black-Scholes-Merton
    - greeks: Calcul des Greeks (Delta, Gamma, Vega, Theta, Rho)
    - implied_volatility: Calcul de la volatilité implicite
    - volatility_smile: Analyse du volatility smile
    - delta_hedging: Simulation de couverture Delta-neutre
"""

from .black_scholes import black_scholes_price, d1, d2
from .greeks import calculate_all_greeks, delta, gamma, vega, theta, rho
from .implied_volatility import implied_volatility_newton
from .volatility_smile import analyze_volatility_smile
from .delta_hedging import simulate_delta_hedging

__version__ = "1.0.0"
__author__ = "M2 Finance Quantitative - Sorbonne"
