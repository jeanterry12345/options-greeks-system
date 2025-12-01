# -*- coding: utf-8 -*-
"""
Calcul des Greeks
=================
Sensibilités des options aux différents paramètres de marché.

Référence: Hull, Chapter 19 - The Greek Letters

Les Greeks mesurent la sensibilité du prix de l'option:
    - Delta (Δ): sensibilité au prix du sous-jacent
    - Gamma (Γ): sensibilité du Delta au prix du sous-jacent
    - Vega (ν): sensibilité à la volatilité
    - Theta (Θ): sensibilité au temps (time decay)
    - Rho (ρ): sensibilité au taux d'intérêt
"""

import numpy as np
from scipy.stats import norm
from .black_scholes import d1, d2


def delta(S, K, T, r, sigma, option_type='call'):
    """
    Calcule le Delta de l'option.

    Le Delta mesure le changement du prix de l'option pour une
    variation de 1€ du prix du sous-jacent.

    Formules (Hull, Section 19.4):
        Call: Δ = N(d₁)
        Put:  Δ = N(d₁) - 1

    Paramètres:
    -----------
    S, K, T, r, sigma : paramètres standard BSM
    option_type : 'call' ou 'put'

    Retourne:
    ---------
    float : Delta (entre 0 et 1 pour call, -1 et 0 pour put)

    Interprétation:
        - Delta = 0.5 signifie que pour +1€ de S, l'option gagne +0.50€
        - Un Delta de 0.5 signifie aussi ~50% de probabilité d'exercice
    """
    d_1 = d1(S, K, T, r, sigma)

    if option_type.lower() == 'call':
        return norm.cdf(d_1)
    else:
        return norm.cdf(d_1) - 1


def gamma(S, K, T, r, sigma):
    """
    Calcule le Gamma de l'option.

    Le Gamma mesure le taux de changement du Delta par rapport
    au prix du sous-jacent. Identique pour call et put.

    Formule (Hull, Section 19.6):
        Γ = N'(d₁) / (S * σ * √T)

    où N'(d₁) est la densité de probabilité normale standard.

    Paramètres:
    -----------
    S, K, T, r, sigma : paramètres standard BSM

    Retourne:
    ---------
    float : Gamma (toujours positif)

    Interprétation:
        - Un Gamma élevé signifie que le Delta change rapidement
        - Gamma est maximal pour les options ATM proches de l'expiration
    """
    d_1 = d1(S, K, T, r, sigma)

    # N'(d₁) = densité normale standard
    n_d1 = norm.pdf(d_1)

    return n_d1 / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """
    Calcule le Vega de l'option.

    Le Vega mesure la sensibilité du prix à un changement de 1%
    de la volatilité. Identique pour call et put.

    Formule (Hull, Section 19.7):
        ν = S * √T * N'(d₁)

    Note: Le Vega est souvent exprimé pour un changement de 1%
    (0.01) de volatilité, donc on divise par 100.

    Paramètres:
    -----------
    S, K, T, r, sigma : paramètres standard BSM

    Retourne:
    ---------
    float : Vega (pour 1% de changement de volatilité)

    Interprétation:
        - Vega = 15 signifie que +1% de volatilité → +0.15€ sur l'option
        - Vega est maximal pour les options ATM à longue maturité
    """
    d_1 = d1(S, K, T, r, sigma)
    n_d1 = norm.pdf(d_1)

    # Vega pour 1% de changement (divisé par 100)
    return S * np.sqrt(T) * n_d1 / 100


def theta(S, K, T, r, sigma, option_type='call'):
    """
    Calcule le Theta de l'option.

    Le Theta mesure la perte de valeur due au passage du temps
    (time decay). Généralement exprimé par jour.

    Formules (Hull, Section 19.5):
        Call: Θ = -[S*N'(d₁)*σ/(2√T)] - r*K*e^(-rT)*N(d₂)
        Put:  Θ = -[S*N'(d₁)*σ/(2√T)] + r*K*e^(-rT)*N(-d₂)

    Paramètres:
    -----------
    S, K, T, r, sigma : paramètres standard BSM
    option_type : 'call' ou 'put'

    Retourne:
    ---------
    float : Theta par jour (généralement négatif)

    Interprétation:
        - Theta = -0.05 signifie que l'option perd 0.05€ par jour
        - Theta s'accélère proche de l'expiration (surtout ATM)
    """
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    n_d1 = norm.pdf(d_1)

    # Premier terme (identique pour call et put)
    first_term = -(S * n_d1 * sigma) / (2 * np.sqrt(T))

    if option_type.lower() == 'call':
        # Second terme pour le call
        second_term = -r * K * np.exp(-r * T) * norm.cdf(d_2)
    else:
        # Second terme pour le put
        second_term = r * K * np.exp(-r * T) * norm.cdf(-d_2)

    # Theta annualisé, divisé par 365 pour avoir par jour
    theta_annual = first_term + second_term
    theta_daily = theta_annual / 365

    return theta_daily


def rho(S, K, T, r, sigma, option_type='call'):
    """
    Calcule le Rho de l'option.

    Le Rho mesure la sensibilité du prix à un changement de 1%
    du taux d'intérêt sans risque.

    Formules (Hull, Section 19.8):
        Call: ρ = K * T * e^(-rT) * N(d₂)
        Put:  ρ = -K * T * e^(-rT) * N(-d₂)

    Paramètres:
    -----------
    S, K, T, r, sigma : paramètres standard BSM
    option_type : 'call' ou 'put'

    Retourne:
    ---------
    float : Rho (pour 1% de changement de taux)

    Interprétation:
        - Call: Rho positif (hausse des taux → hausse du call)
        - Put: Rho négatif (hausse des taux → baisse du put)
    """
    d_2 = d2(S, K, T, r, sigma)

    if option_type.lower() == 'call':
        rho_value = K * T * np.exp(-r * T) * norm.cdf(d_2)
    else:
        rho_value = -K * T * np.exp(-r * T) * norm.cdf(-d_2)

    # Rho pour 1% de changement (divisé par 100)
    return rho_value / 100


def calculate_all_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calcule tous les Greeks d'une option.

    Paramètres:
    -----------
    S, K, T, r, sigma : paramètres standard BSM
    option_type : 'call' ou 'put'

    Retourne:
    ---------
    dict : dictionnaire contenant tous les Greeks

    Exemple:
    --------
    >>> greeks = calculate_all_greeks(100, 100, 1, 0.05, 0.2, 'call')
    >>> print(f"Delta: {greeks['delta']:.4f}")
    """
    return {
        'delta': delta(S, K, T, r, sigma, option_type),
        'gamma': gamma(S, K, T, r, sigma),
        'vega': vega(S, K, T, r, sigma),
        'theta': theta(S, K, T, r, sigma, option_type),
        'rho': rho(S, K, T, r, sigma, option_type)
    }


def print_greeks(greeks, option_type='call'):
    """
    Affiche les Greeks de manière formatée.

    Paramètres:
    -----------
    greeks : dict
        Dictionnaire des Greeks (retourné par calculate_all_greeks)
    option_type : str
        Type d'option pour l'affichage
    """
    print(f"\n=== Greeks du {option_type.upper()} ===")
    print(f"Delta : {greeks['delta']:+.4f}")
    print(f"Gamma : {greeks['gamma']:.6f}")
    print(f"Vega  : {greeks['vega']:.4f} (pour +1% vol)")
    print(f"Theta : {greeks['theta']:.4f} €/jour")
    print(f"Rho   : {greeks['rho']:.4f} (pour +1% taux)")


# Tests si exécuté directement
if __name__ == "__main__":
    # Paramètres de test
    S = 100     # Prix spot
    K = 100     # Strike (ATM)
    T = 1.0     # 1 an
    r = 0.05    # 5% taux sans risque
    sigma = 0.2 # 20% volatilité

    print("=== Test des Greeks ===")
    print(f"S = {S}, K = {K}, T = {T}, r = {r}, σ = {sigma}")

    # Greeks du Call
    call_greeks = calculate_all_greeks(S, K, T, r, sigma, 'call')
    print_greeks(call_greeks, 'call')

    # Greeks du Put
    put_greeks = calculate_all_greeks(S, K, T, r, sigma, 'put')
    print_greeks(put_greeks, 'put')

    # Vérification: Delta(call) - Delta(put) = 1
    print(f"\nVérification: Delta(call) - Delta(put) = {call_greeks['delta'] - put_greeks['delta']:.4f} (devrait être ~1)")
