# -*- coding: utf-8 -*-
"""
Modèle Black-Scholes-Merton
===========================
Implémentation du modèle BSM pour le pricing d'options européennes.

Référence: Hull, Chapter 15 - The Black-Scholes-Merton Model

Formules principales (Hull, Eq. 15.20 et 15.21):
    c = S₀ * N(d₁) - K * e^(-rT) * N(d₂)
    p = K * e^(-rT) * N(-d₂) - S₀ * N(-d₁)

où:
    d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
    d₂ = d₁ - σ√T
"""

import numpy as np
from scipy.stats import norm


def d1(S, K, T, r, sigma):
    """
    Calcule d₁ dans la formule Black-Scholes.

    Paramètres:
    -----------
    S : float
        Prix spot de l'actif sous-jacent
    K : float
        Prix d'exercice (strike)
    T : float
        Temps jusqu'à maturité (en années)
    r : float
        Taux d'intérêt sans risque (annualisé)
    sigma : float
        Volatilité de l'actif (annualisée)

    Retourne:
    ---------
    float : valeur de d₁

    Formule (Hull, Eq. 15.20):
        d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    """
    # Vérification que T > 0 pour éviter division par zéro
    if T <= 0:
        return 0.0

    numerator = np.log(S / K) + (r + 0.5 * sigma ** 2) * T
    denominator = sigma * np.sqrt(T)

    return numerator / denominator


def d2(S, K, T, r, sigma):
    """
    Calcule d₂ dans la formule Black-Scholes.

    Paramètres:
    -----------
    S, K, T, r, sigma : voir d1()

    Retourne:
    ---------
    float : valeur de d₂

    Formule (Hull, Eq. 15.20):
        d₂ = d₁ - σ√T
    """
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Calcule le prix d'une option européenne avec Black-Scholes.

    Paramètres:
    -----------
    S : float
        Prix spot de l'actif sous-jacent
    K : float
        Prix d'exercice (strike)
    T : float
        Temps jusqu'à maturité (en années)
    r : float
        Taux d'intérêt sans risque (annualisé)
    sigma : float
        Volatilité de l'actif (annualisée)
    option_type : str
        'call' ou 'put'

    Retourne:
    ---------
    float : prix de l'option

    Exemple:
    --------
    >>> price = black_scholes_price(100, 100, 1, 0.05, 0.2, 'call')
    >>> print(f"Prix du Call: {price:.4f}")
    Prix du Call: 10.4506
    """
    # Calcul de d1 et d2
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)

    # Facteur d'actualisation
    discount = np.exp(-r * T)

    if option_type.lower() == 'call':
        # Prix du Call (Hull, Eq. 15.20)
        # c = S * N(d₁) - K * e^(-rT) * N(d₂)
        price = S * norm.cdf(d_1) - K * discount * norm.cdf(d_2)

    elif option_type.lower() == 'put':
        # Prix du Put (Hull, Eq. 15.21)
        # p = K * e^(-rT) * N(-d₂) - S * N(-d₁)
        price = K * discount * norm.cdf(-d_2) - S * norm.cdf(-d_1)

    else:
        raise ValueError("option_type doit être 'call' ou 'put'")

    return price


def put_call_parity_check(S, K, T, r, call_price, put_price):
    """
    Vérifie la parité Put-Call.

    Formule (Hull, Eq. 11.6):
        c + K*e^(-rT) = p + S

    Paramètres:
    -----------
    S, K, T, r : voir black_scholes_price()
    call_price : float
        Prix observé du Call
    put_price : float
        Prix observé du Put

    Retourne:
    ---------
    dict : contenant l'écart et si la parité est respectée
    """
    # Côté gauche: Call + valeur actualisée du strike
    left_side = call_price + K * np.exp(-r * T)

    # Côté droit: Put + prix spot
    right_side = put_price + S

    # Écart
    difference = abs(left_side - right_side)

    # Tolérance de 1% du prix spot
    tolerance = 0.01 * S
    is_valid = difference < tolerance

    return {
        'left_side': left_side,
        'right_side': right_side,
        'difference': difference,
        'is_valid': is_valid,
        'message': "Parité respectée" if is_valid else "Violation de la parité Put-Call"
    }


# Tests rapides si exécuté directement
if __name__ == "__main__":
    # Exemple du livre de Hull
    S = 42      # Prix spot
    K = 40      # Strike
    T = 0.5     # 6 mois
    r = 0.10    # 10% taux sans risque
    sigma = 0.2 # 20% volatilité

    print("=== Test Black-Scholes ===")
    print(f"S = {S}, K = {K}, T = {T}, r = {r}, σ = {sigma}")
    print()

    # Calcul des prix
    call_price = black_scholes_price(S, K, T, r, sigma, 'call')
    put_price = black_scholes_price(S, K, T, r, sigma, 'put')

    print(f"Prix Call : {call_price:.4f} €")
    print(f"Prix Put  : {put_price:.4f} €")
    print()

    # Vérification parité Put-Call
    parity = put_call_parity_check(S, K, T, r, call_price, put_price)
    print(f"Parité Put-Call : {parity['message']}")
    print(f"Écart : {parity['difference']:.6f}")
