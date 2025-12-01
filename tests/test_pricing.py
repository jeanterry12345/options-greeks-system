# -*- coding: utf-8 -*-
"""
Tests Unitaires pour le Système de Pricing d'Options
====================================================
Tests basiques pour valider les calculs Black-Scholes et Greeks.
"""

import sys
import os
import numpy as np

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.black_scholes import black_scholes_price, d1, d2, put_call_parity_check
from src.greeks import delta, gamma, vega, theta, rho, calculate_all_greeks
from src.implied_volatility import implied_volatility_newton, calculate_implied_vol


def test_black_scholes_call():
    """
    Test du prix d'un Call avec les valeurs de référence de Hull.

    Exemple Hull: S=42, K=40, T=0.5, r=0.10, σ=0.20
    Prix attendu du Call ≈ 4.76
    """
    S, K, T, r, sigma = 42, 40, 0.5, 0.10, 0.20

    call_price = black_scholes_price(S, K, T, r, sigma, 'call')

    # Valeur de référence (Hull, Example 15.6)
    expected = 4.76

    assert abs(call_price - expected) < 0.1, \
        f"Prix Call incorrect: {call_price:.4f}, attendu: {expected}"

    print(f"✓ Test Black-Scholes Call: {call_price:.4f} (attendu: ~{expected})")


def test_black_scholes_put():
    """
    Test du prix d'un Put.
    """
    S, K, T, r, sigma = 42, 40, 0.5, 0.10, 0.20

    put_price = black_scholes_price(S, K, T, r, sigma, 'put')

    # Valeur attendue calculée avec la parité Put-Call
    call_price = black_scholes_price(S, K, T, r, sigma, 'call')
    expected_put = call_price + K * np.exp(-r * T) - S

    assert abs(put_price - expected_put) < 0.001, \
        f"Prix Put incorrect: {put_price:.4f}, attendu: {expected_put:.4f}"

    print(f"✓ Test Black-Scholes Put: {put_price:.4f}")


def test_put_call_parity():
    """
    Test de la parité Put-Call: c + K*e^(-rT) = p + S
    """
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    call = black_scholes_price(S, K, T, r, sigma, 'call')
    put = black_scholes_price(S, K, T, r, sigma, 'put')

    parity = put_call_parity_check(S, K, T, r, call, put)

    assert parity['is_valid'], \
        f"Parité Put-Call violée: écart = {parity['difference']:.6f}"

    print(f"✓ Test Parité Put-Call: écart = {parity['difference']:.2e}")


def test_delta_bounds():
    """
    Test des bornes du Delta:
        - Call: 0 ≤ Delta ≤ 1
        - Put: -1 ≤ Delta ≤ 0
    """
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    call_delta = delta(S, K, T, r, sigma, 'call')
    put_delta = delta(S, K, T, r, sigma, 'put')

    assert 0 <= call_delta <= 1, f"Delta Call hors bornes: {call_delta}"
    assert -1 <= put_delta <= 0, f"Delta Put hors bornes: {put_delta}"

    # Delta(call) - Delta(put) = 1
    delta_diff = call_delta - put_delta
    assert abs(delta_diff - 1) < 0.001, \
        f"Delta(call) - Delta(put) ≠ 1: {delta_diff}"

    print(f"✓ Test Delta: Call={call_delta:.4f}, Put={put_delta:.4f}")


def test_gamma_positive():
    """
    Test que Gamma est toujours positif (call et put identiques).
    """
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    g = gamma(S, K, T, r, sigma)

    assert g > 0, f"Gamma négatif: {g}"

    print(f"✓ Test Gamma: {g:.6f} (positif)")


def test_vega_positive():
    """
    Test que Vega est toujours positif.
    """
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    v = vega(S, K, T, r, sigma)

    assert v > 0, f"Vega négatif: {v}"

    print(f"✓ Test Vega: {v:.4f} (positif)")


def test_theta_negative_for_atm():
    """
    Test que Theta est généralement négatif pour une option ATM.
    (Time decay - l'option perd de la valeur avec le temps)
    """
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    call_theta = theta(S, K, T, r, sigma, 'call')

    # Theta du Call ATM devrait être négatif
    assert call_theta < 0, f"Theta Call ATM positif: {call_theta}"

    print(f"✓ Test Theta: {call_theta:.4f} (négatif pour ATM)")


def test_implied_volatility():
    """
    Test de la volatilité implicite: on calcule un prix avec σ connue,
    puis on retrouve σ à partir du prix.
    """
    S, K, T, r = 100, 100, 1.0, 0.05
    true_sigma = 0.25

    # Calculer le prix théorique
    price = black_scholes_price(S, K, T, r, true_sigma, 'call')

    # Retrouver la volatilité implicite
    result = implied_volatility_newton(price, S, K, T, r, 'call')

    assert result['converged'], "Newton-Raphson n'a pas convergé"
    assert abs(result['implied_vol'] - true_sigma) < 0.001, \
        f"Vol implicite incorrecte: {result['implied_vol']:.4f}, attendu: {true_sigma}"

    print(f"✓ Test Vol Implicite: {result['implied_vol']:.4f} (attendu: {true_sigma})")


def test_all_greeks_consistency():
    """
    Test de cohérence des Greeks calculés ensemble.
    """
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    greeks = calculate_all_greeks(S, K, T, r, sigma, 'call')

    # Vérifier que tous les Greeks sont des nombres finis
    for greek_name, value in greeks.items():
        assert np.isfinite(value), f"{greek_name} n'est pas fini: {value}"

    print(f"✓ Test cohérence Greeks: tous les valeurs sont finies")


def test_deep_itm_call():
    """
    Test d'un Call deep ITM: prix ≈ S - K*e^(-rT)
    """
    S, K, T, r, sigma = 150, 100, 1.0, 0.05, 0.2

    call_price = black_scholes_price(S, K, T, r, sigma, 'call')

    # Prix minimum = valeur intrinsèque actualisée
    intrinsic = S - K * np.exp(-r * T)

    assert call_price >= intrinsic, \
        f"Prix Call < valeur intrinsèque: {call_price:.4f} < {intrinsic:.4f}"

    print(f"✓ Test Deep ITM Call: prix={call_price:.4f} ≥ intrinsèque={intrinsic:.4f}")


def test_deep_otm_call():
    """
    Test d'un Call deep OTM: prix proche de 0.
    """
    S, K, T, r, sigma = 50, 100, 1.0, 0.05, 0.2

    call_price = black_scholes_price(S, K, T, r, sigma, 'call')

    assert call_price < 1.0, \
        f"Prix Call OTM trop élevé: {call_price:.4f}"
    assert call_price >= 0, \
        f"Prix Call négatif: {call_price:.4f}"

    print(f"✓ Test Deep OTM Call: prix={call_price:.4f} (proche de 0)")


def run_all_tests():
    """
    Exécute tous les tests unitaires.
    """
    print("=" * 50)
    print("TESTS UNITAIRES - Système de Pricing d'Options")
    print("=" * 50)
    print()

    tests = [
        test_black_scholes_call,
        test_black_scholes_put,
        test_put_call_parity,
        test_delta_bounds,
        test_gamma_positive,
        test_vega_positive,
        test_theta_negative_for_atm,
        test_implied_volatility,
        test_all_greeks_consistency,
        test_deep_itm_call,
        test_deep_otm_call
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} ÉCHEC: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERREUR: {e}")
            failed += 1

    print()
    print("=" * 50)
    print(f"Résultats: {passed} réussis, {failed} échoués sur {len(tests)}")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
