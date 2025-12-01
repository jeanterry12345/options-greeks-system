# -*- coding: utf-8 -*-
"""
Démonstration du Système de Pricing d'Options
=============================================
Script de démonstration utilisant des données réelles du CAC40.

Ce script montre:
1. Pricing Black-Scholes avec données de marché
2. Calcul des Greeks
3. Calcul de la volatilité implicite
4. Analyse du volatility smile
5. Simulation de Delta hedging
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.black_scholes import black_scholes_price, put_call_parity_check
from src.greeks import calculate_all_greeks, print_greeks
from src.implied_volatility import implied_volatility_newton, calculate_implied_vol
from src.volatility_smile import (analyze_volatility_smile, plot_volatility_smile,
                                   calculate_smile_metrics, generate_synthetic_smile_data)
from src.delta_hedging import simulate_delta_hedging, plot_hedging_simulation


def demo_pricing_basic():
    """
    Démonstration 1: Pricing basique Black-Scholes.
    """
    print("\n" + "=" * 60)
    print("DÉMONSTRATION 1: PRICING BLACK-SCHOLES")
    print("=" * 60)

    # Paramètres basés sur une action française typique (ex: TotalEnergies)
    # Ces valeurs sont à titre d'exemple
    S = 58.50       # Prix spot TotalEnergies (exemple)
    K = 60.00       # Strike
    T = 0.25        # 3 mois (0.25 an)
    r = 0.035       # Taux EUR ~3.5%
    sigma = 0.22    # Volatilité historique ~22%

    print(f"\nParamètres du marché:")
    print(f"  Sous-jacent (S): {S:.2f} €")
    print(f"  Strike (K): {K:.2f} €")
    print(f"  Maturité (T): {T:.2f} ans ({int(T*12)} mois)")
    print(f"  Taux sans risque (r): {r:.2%}")
    print(f"  Volatilité (σ): {sigma:.2%}")

    # Calcul des prix
    call_price = black_scholes_price(S, K, T, r, sigma, 'call')
    put_price = black_scholes_price(S, K, T, r, sigma, 'put')

    print(f"\n=== Prix des Options ===")
    print(f"Prix Call: {call_price:.4f} €")
    print(f"Prix Put:  {put_price:.4f} €")

    # Vérification parité Put-Call
    parity = put_call_parity_check(S, K, T, r, call_price, put_price)
    print(f"\nParité Put-Call: {parity['message']}")


def demo_greeks():
    """
    Démonstration 2: Calcul et interprétation des Greeks.
    """
    print("\n" + "=" * 60)
    print("DÉMONSTRATION 2: LES GREEKS")
    print("=" * 60)

    S = 100         # Prix spot (option ATM)
    K = 100         # Strike
    T = 1.0         # 1 an
    r = 0.05        # 5%
    sigma = 0.20    # 20%

    print(f"\nParamètres: S={S}, K={K}, T={T}, r={r}, σ={sigma}")

    # Greeks du Call
    call_greeks = calculate_all_greeks(S, K, T, r, sigma, 'call')
    print_greeks(call_greeks, 'call')

    print("\n--- Interprétation ---")
    print(f"• Delta = {call_greeks['delta']:.2f}: Si S ↑ de 1€, le Call ↑ de {call_greeks['delta']:.2f}€")
    print(f"• Gamma = {call_greeks['gamma']:.4f}: Le Delta change de {call_greeks['gamma']:.4f} pour +1€ de S")
    print(f"• Vega = {call_greeks['vega']:.2f}: Si σ ↑ de 1%, le Call ↑ de {call_greeks['vega']:.2f}€")
    print(f"• Theta = {call_greeks['theta']:.4f}: Le Call perd {abs(call_greeks['theta']):.4f}€ par jour")
    print(f"• Rho = {call_greeks['rho']:.4f}: Si r ↑ de 1%, le Call ↑ de {call_greeks['rho']:.4f}€")


def demo_implied_volatility():
    """
    Démonstration 3: Calcul de la volatilité implicite.
    """
    print("\n" + "=" * 60)
    print("DÉMONSTRATION 3: VOLATILITÉ IMPLICITE")
    print("=" * 60)

    # Simulation de prix de marché observés
    S = 100
    K_values = [90, 95, 100, 105, 110]
    T = 0.5
    r = 0.05

    # Prix de marché simulés (avec un smile)
    market_prices = [12.50, 9.20, 6.80, 4.90, 3.50]

    print(f"\nExtraction de la volatilité implicite:")
    print(f"{'Strike':<10} {'Prix Marché':<15} {'Vol Implicite':<15} {'Moneyness':<10}")
    print("-" * 50)

    for K, price in zip(K_values, market_prices):
        iv = calculate_implied_vol(price, S, K, T, r, 'call')
        moneyness = K / S
        if not np.isnan(iv):
            print(f"{K:<10} {price:<15.2f} {iv*100:<15.2f}% {moneyness:<10.2f}")
        else:
            print(f"{K:<10} {price:<15.2f} {'N/A':<15} {moneyness:<10.2f}")


def demo_volatility_smile():
    """
    Démonstration 4: Analyse du volatility smile.
    """
    print("\n" + "=" * 60)
    print("DÉMONSTRATION 4: VOLATILITY SMILE")
    print("=" * 60)

    S = 100
    T = 0.5
    r = 0.05

    # Générer des données synthétiques avec un smirk typique des actions
    print("\nGénération de données synthétiques avec smirk:")
    option_data = generate_synthetic_smile_data(
        S=S, T=T, r=r,
        base_vol=0.20,      # Vol ATM = 20%
        skew=-0.15,         # Smirk négatif
        convexity=0.08      # Convexité modérée
    )

    # Analyser le smile
    df = analyze_volatility_smile(option_data, S, T, r)

    print("\nDonnées du smile:")
    print(df[['strike', 'moneyness', 'implied_vol']].to_string(index=False))

    # Métriques
    metrics = calculate_smile_metrics(df)
    print(f"\n=== Métriques du Smile ===")
    print(f"Vol ATM: {metrics['atm_vol']:.2%}")
    print(f"Range de vol: {metrics['vol_range']:.2%}")

    # Tracer le graphique
    fig = plot_volatility_smile(df, "Volatility Smile - Données Synthétiques")
    plt.savefig('demo_volatility_smile.png', dpi=150, bbox_inches='tight')
    print("\nGraphique sauvegardé: demo_volatility_smile.png")
    plt.close()


def demo_delta_hedging():
    """
    Démonstration 5: Simulation de couverture Delta-neutre.
    """
    print("\n" + "=" * 60)
    print("DÉMONSTRATION 5: DELTA HEDGING")
    print("=" * 60)

    S0 = 100
    K = 100
    T = 0.5     # 6 mois
    r = 0.05
    sigma = 0.20

    print(f"\nSimulation d'une couverture Delta-neutre:")
    print(f"Position: Vente d'un Call ATM")
    print(f"Paramètres: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")

    # Simulation avec 126 rebalancements (quotidien sur 6 mois)
    df, summary = simulate_delta_hedging(S0, K, T, r, sigma, n_steps=126, seed=42)

    print(f"\n=== Résultats ===")
    print(f"Prime reçue: {summary['option_premium']:.4f} €")
    print(f"Prix final S: {summary['final_stock_price']:.2f} €")
    print(f"Payoff option: {summary['option_payoff']:.4f} €")
    print(f"Erreur de hedging: {summary['hedging_error']:.4f} € ({summary['error_percent']:.2f}%)")

    # Tracer les graphiques
    fig = plot_hedging_simulation(df, summary, "Delta Hedging - Simulation")
    plt.savefig('demo_delta_hedging.png', dpi=150, bbox_inches='tight')
    print("\nGraphique sauvegardé: demo_delta_hedging.png")
    plt.close()


def demo_market_comparison():
    """
    Démonstration 6: Comparaison avec données de marché simulées.

    Note: En production, on utiliserait yfinance pour récupérer
    les vraies données d'options CAC40.
    """
    print("\n" + "=" * 60)
    print("DÉMONSTRATION 6: COMPARAISON AVEC LE MARCHÉ")
    print("=" * 60)

    # Données simulées pour 5 actions CAC40
    stocks = {
        'TotalEnergies': {'S': 58.50, 'sigma_hist': 0.22},
        'LVMH': {'S': 750.00, 'sigma_hist': 0.25},
        'Airbus': {'S': 135.00, 'sigma_hist': 0.28},
        'Sanofi': {'S': 92.00, 'sigma_hist': 0.18},
        'BNP Paribas': {'S': 62.00, 'sigma_hist': 0.30}
    }

    T = 0.25  # 3 mois
    r = 0.035 # Taux EUR

    print(f"\nPricing d'options ATM sur actions CAC40 (T={T} ans, r={r:.2%}):")
    print(f"{'Action':<15} {'Spot':<10} {'Call ATM':<12} {'Put ATM':<12} {'Vol':<10}")
    print("-" * 60)

    for name, data in stocks.items():
        S = data['S']
        K = round(S)  # Strike arrondi (ATM)
        sigma = data['sigma_hist']

        call = black_scholes_price(S, K, T, r, sigma, 'call')
        put = black_scholes_price(S, K, T, r, sigma, 'put')

        print(f"{name:<15} {S:<10.2f} {call:<12.2f} {put:<12.2f} {sigma*100:<10.1f}%")


def main():
    """
    Exécute toutes les démonstrations.
    """
    print("\n" + "#" * 60)
    print("#" + " " * 15 + "SYSTÈME DE PRICING D'OPTIONS" + " " * 15 + "#")
    print("#" + " " * 12 + "Démonstration des Fonctionnalités" + " " * 11 + "#")
    print("#" * 60)

    demo_pricing_basic()
    demo_greeks()
    demo_implied_volatility()
    demo_volatility_smile()
    demo_delta_hedging()
    demo_market_comparison()

    print("\n" + "=" * 60)
    print("FIN DE LA DÉMONSTRATION")
    print("=" * 60)
    print("\nFichiers générés:")
    print("  - demo_volatility_smile.png")
    print("  - demo_delta_hedging.png")


if __name__ == "__main__":
    main()
