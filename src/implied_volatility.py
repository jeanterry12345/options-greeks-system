# -*- coding: utf-8 -*-
"""
Calcul de la Volatilité Implicite
=================================
Méthode Newton-Raphson pour extraire la volatilité implicite
à partir des prix d'options observés sur le marché.

Référence: Hull, Chapter 20 - Volatility Smiles

La volatilité implicite est la valeur de σ qui, injectée dans
la formule Black-Scholes, donne le prix de marché observé.
"""

import numpy as np
from .black_scholes import black_scholes_price
from .greeks import vega


def implied_volatility_newton(market_price, S, K, T, r, option_type='call',
                               initial_vol=0.2, tolerance=1e-6, max_iterations=100):
    """
    Calcule la volatilité implicite par la méthode Newton-Raphson.

    La méthode Newton-Raphson utilise la formule itérative:
        σ_{n+1} = σ_n - f(σ_n) / f'(σ_n)

    où:
        f(σ) = Prix_BS(σ) - Prix_marché
        f'(σ) = Vega (dérivée du prix par rapport à σ)

    Paramètres:
    -----------
    market_price : float
        Prix de l'option observé sur le marché
    S : float
        Prix spot du sous-jacent
    K : float
        Prix d'exercice (strike)
    T : float
        Temps jusqu'à maturité (en années)
    r : float
        Taux sans risque
    option_type : str
        'call' ou 'put'
    initial_vol : float
        Volatilité initiale pour démarrer l'algorithme (défaut: 20%)
    tolerance : float
        Critère de convergence (défaut: 1e-6)
    max_iterations : int
        Nombre maximum d'itérations (défaut: 100)

    Retourne:
    ---------
    dict : contenant la volatilité implicite et les informations de convergence

    Exemple:
    --------
    >>> result = implied_volatility_newton(10.45, 100, 100, 1, 0.05, 'call')
    >>> print(f"Vol implicite: {result['implied_vol']:.2%}")
    Vol implicite: 20.00%
    """
    sigma = initial_vol

    for i in range(max_iterations):
        # Prix théorique avec la volatilité actuelle
        bs_price = black_scholes_price(S, K, T, r, sigma, option_type)

        # Différence avec le prix de marché
        price_diff = bs_price - market_price

        # Vérifier la convergence
        if abs(price_diff) < tolerance:
            return {
                'implied_vol': sigma,
                'converged': True,
                'iterations': i + 1,
                'final_error': abs(price_diff)
            }

        # Calcul du Vega (attention: notre fonction vega est déjà /100)
        option_vega = vega(S, K, T, r, sigma) * 100  # Remettre à l'échelle

        # Éviter la division par zéro
        if abs(option_vega) < 1e-10:
            break

        # Mise à jour Newton-Raphson
        sigma = sigma - price_diff / option_vega

        # S'assurer que sigma reste positif
        if sigma <= 0:
            sigma = 0.01

    # Pas de convergence
    return {
        'implied_vol': sigma,
        'converged': False,
        'iterations': max_iterations,
        'final_error': abs(black_scholes_price(S, K, T, r, sigma, option_type) - market_price)
    }


def implied_volatility_bisection(market_price, S, K, T, r, option_type='call',
                                  vol_low=0.001, vol_high=3.0, tolerance=1e-6,
                                  max_iterations=100):
    """
    Calcule la volatilité implicite par dichotomie (bisection).

    Méthode plus robuste que Newton-Raphson, mais plus lente.
    Utilisée comme backup si Newton-Raphson ne converge pas.

    Principe:
        1. On part d'un intervalle [vol_low, vol_high]
        2. On calcule le prix au milieu
        3. On réduit l'intervalle de moitié selon le signe de l'erreur
        4. On répète jusqu'à convergence

    Paramètres:
    -----------
    market_price : float
        Prix de marché de l'option
    S, K, T, r : paramètres BSM standard
    option_type : str
        'call' ou 'put'
    vol_low, vol_high : float
        Bornes de l'intervalle de recherche
    tolerance, max_iterations : paramètres de convergence

    Retourne:
    ---------
    dict : volatilité implicite et informations de convergence
    """
    for i in range(max_iterations):
        vol_mid = (vol_low + vol_high) / 2

        # Prix au milieu
        price_mid = black_scholes_price(S, K, T, r, vol_mid, option_type)
        diff = price_mid - market_price

        # Vérifier la convergence
        if abs(diff) < tolerance:
            return {
                'implied_vol': vol_mid,
                'converged': True,
                'iterations': i + 1,
                'final_error': abs(diff)
            }

        # Réduire l'intervalle
        if diff > 0:
            # Prix trop élevé, réduire la volatilité
            vol_high = vol_mid
        else:
            # Prix trop bas, augmenter la volatilité
            vol_low = vol_mid

    return {
        'implied_vol': vol_mid,
        'converged': False,
        'iterations': max_iterations,
        'final_error': abs(black_scholes_price(S, K, T, r, vol_mid, option_type) - market_price)
    }


def calculate_implied_vol(market_price, S, K, T, r, option_type='call'):
    """
    Fonction principale pour calculer la volatilité implicite.

    Essaie d'abord Newton-Raphson, puis bisection si échec.

    Paramètres:
    -----------
    market_price : float
        Prix de marché de l'option
    S, K, T, r : paramètres BSM standard
    option_type : str
        'call' ou 'put'

    Retourne:
    ---------
    float : volatilité implicite (ou NaN si échec)
    """
    # Essayer Newton-Raphson d'abord
    result = implied_volatility_newton(market_price, S, K, T, r, option_type)

    if result['converged']:
        return result['implied_vol']

    # Si échec, essayer bisection
    result = implied_volatility_bisection(market_price, S, K, T, r, option_type)

    if result['converged']:
        return result['implied_vol']

    # Échec total
    return np.nan


# Tests si exécuté directement
if __name__ == "__main__":
    from black_scholes import black_scholes_price

    # Paramètres de test
    S = 100
    K = 100
    T = 1.0
    r = 0.05
    true_sigma = 0.25  # Vraie volatilité

    # Calculer le prix théorique
    market_price = black_scholes_price(S, K, T, r, true_sigma, 'call')
    print(f"=== Test Volatilité Implicite ===")
    print(f"Prix de marché (σ=25%): {market_price:.4f}")

    # Retrouver la volatilité implicite
    result = implied_volatility_newton(market_price, S, K, T, r, 'call')

    print(f"\nRésultat Newton-Raphson:")
    print(f"  Vol implicite: {result['implied_vol']:.2%}")
    print(f"  Convergence: {result['converged']}")
    print(f"  Itérations: {result['iterations']}")
    print(f"  Erreur finale: {result['final_error']:.2e}")

    # Vérification
    ecart = abs(result['implied_vol'] - true_sigma)
    print(f"\nÉcart avec vraie volatilité: {ecart:.2e}")
