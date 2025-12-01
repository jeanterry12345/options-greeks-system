# -*- coding: utf-8 -*-
"""
Couverture Delta-Neutre (Delta Hedging)
=======================================
Simulation de stratégie de couverture dynamique pour les options.

Référence: Hull, Chapter 19 - The Greek Letters (Section 19.4)

Le Delta hedging consiste à maintenir un portefeuille Delta-neutre
en ajustant continuellement la position sur le sous-jacent.

Principe:
    - Vendeur d'option: achète Delta * S d'actions pour couvrir
    - Rebalancement périodique quand le Delta change
    - Coût de hedging ≈ prix de l'option (en théorie BSM)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .black_scholes import black_scholes_price
from .greeks import delta


def simulate_stock_path(S0, r, sigma, T, n_steps, seed=None):
    """
    Simule un chemin de prix du sous-jacent (mouvement brownien géométrique).

    Modèle (Hull, Eq. 14.14):
        dS = μ*S*dt + σ*S*dW

    En notation discrète:
        S(t+dt) = S(t) * exp((r - σ²/2)*dt + σ*√dt*Z)

    Paramètres:
    -----------
    S0 : float
        Prix initial
    r : float
        Drift (taux sans risque en risque-neutre)
    sigma : float
        Volatilité
    T : float
        Horizon de temps
    n_steps : int
        Nombre de pas de temps
    seed : int
        Graine aléatoire pour reproductibilité

    Retourne:
    ---------
    np.array : chemin de prix de taille (n_steps + 1,)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    path = np.zeros(n_steps + 1)
    path[0] = S0

    # Simulation du mouvement brownien géométrique
    for i in range(1, n_steps + 1):
        # Tirage aléatoire normal
        Z = np.random.standard_normal()

        # Formule exacte (pas de biais de discrétisation)
        path[i] = path[i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    return path


def simulate_delta_hedging(S0, K, T, r, sigma, n_steps=252, position='short_call',
                            seed=None):
    """
    Simule une stratégie de couverture Delta-neutre.

    Scénario: On vend un Call et on se couvre dynamiquement.

    Paramètres:
    -----------
    S0 : float
        Prix spot initial
    K : float
        Strike de l'option
    T : float
        Maturité (en années)
    r : float
        Taux sans risque
    sigma : float
        Volatilité
    n_steps : int
        Nombre de rebalancements (défaut: 252 = quotidien sur 1 an)
    position : str
        'short_call' ou 'short_put'
    seed : int
        Graine aléatoire

    Retourne:
    ---------
    pd.DataFrame : historique complet de la couverture
    dict : résumé de la performance

    Colonnes du DataFrame:
        - time: temps
        - stock_price: prix du sous-jacent
        - delta: delta de l'option
        - shares_held: nombre d'actions détenues
        - option_value: valeur de l'option
        - hedge_value: valeur de la couverture
        - cash: position cash
        - portfolio_value: valeur totale du portefeuille
    """
    dt = T / n_steps
    times = np.linspace(0, T, n_steps + 1)

    # Simuler le chemin de prix
    stock_path = simulate_stock_path(S0, r, sigma, T, n_steps, seed)

    # Déterminer le type d'option
    if 'call' in position.lower():
        option_type = 'call'
    else:
        option_type = 'put'

    # Initialisation des tableaux
    deltas = np.zeros(n_steps + 1)
    shares = np.zeros(n_steps + 1)
    option_values = np.zeros(n_steps + 1)
    cash = np.zeros(n_steps + 1)
    portfolio_values = np.zeros(n_steps + 1)

    # Valeurs initiales
    # On vend l'option et reçoit la prime
    option_premium = black_scholes_price(S0, K, T, r, sigma, option_type)
    deltas[0] = delta(S0, K, T, r, sigma, option_type)
    shares[0] = deltas[0]  # Nombre d'actions à acheter pour couvrir

    # Cash initial = prime reçue - coût des actions achetées
    cash[0] = option_premium - shares[0] * S0

    option_values[0] = option_premium
    portfolio_values[0] = cash[0] + shares[0] * S0 - option_premium

    # Simulation pas à pas
    for i in range(1, n_steps + 1):
        S = stock_path[i]
        t = times[i]
        tau = T - t  # Temps restant

        # Valeur de l'option à cet instant
        if tau > 0.0001:  # Éviter les problèmes numériques proches de l'expiration
            option_values[i] = black_scholes_price(S, K, tau, r, sigma, option_type)
            deltas[i] = delta(S, K, tau, r, sigma, option_type)
        else:
            # À l'expiration
            if option_type == 'call':
                option_values[i] = max(S - K, 0)
                deltas[i] = 1.0 if S > K else 0.0
            else:
                option_values[i] = max(K - S, 0)
                deltas[i] = -1.0 if S < K else 0.0

        # Rebalancement: ajuster le nombre d'actions
        delta_change = deltas[i] - shares[i-1]
        shares[i] = deltas[i]

        # Mise à jour du cash (avec intérêts et coût de transaction)
        # Cash croît au taux sans risque
        interest = cash[i-1] * (np.exp(r * dt) - 1)
        # Coût d'achat/vente des actions
        transaction_cost = delta_change * S

        cash[i] = cash[i-1] + interest - transaction_cost

        # Valeur du portefeuille de couverture
        # = Cash + Actions - Option vendue
        portfolio_values[i] = cash[i] + shares[i] * S - option_values[i]

    # Créer le DataFrame
    df = pd.DataFrame({
        'time': times,
        'stock_price': stock_path,
        'delta': deltas,
        'shares_held': shares,
        'option_value': option_values,
        'cash': cash,
        'portfolio_value': portfolio_values
    })

    # Calculer les métriques de performance
    final_S = stock_path[-1]
    if option_type == 'call':
        option_payoff = max(final_S - K, 0)
    else:
        option_payoff = max(K - final_S, 0)

    # P&L de la couverture
    # Nous avons vendu l'option, donc notre profit = prime - payoff final
    hedge_pnl = option_premium - option_payoff
    portfolio_final = portfolio_values[-1]

    summary = {
        'option_premium': option_premium,
        'final_stock_price': final_S,
        'option_payoff': option_payoff,
        'theoretical_pnl': hedge_pnl,
        'hedge_portfolio_value': portfolio_final,
        'hedging_error': portfolio_final,  # Devrait être proche de 0
        'error_percent': abs(portfolio_final) / option_premium * 100 if option_premium > 0 else 0
    }

    return df, summary


def plot_hedging_simulation(df, summary, title="Simulation Delta Hedging"):
    """
    Trace les graphiques de la simulation de couverture.

    Paramètres:
    -----------
    df : pd.DataFrame
        Résultat de simulate_delta_hedging
    summary : dict
        Résumé de performance
    title : str
        Titre du graphique
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Graphique 1: Prix du sous-jacent
    ax1 = axes[0, 0]
    ax1.plot(df['time'], df['stock_price'], 'b-', linewidth=1.5)
    ax1.set_xlabel('Temps (années)')
    ax1.set_ylabel('Prix du sous-jacent (€)')
    ax1.set_title('Évolution du prix du sous-jacent')
    ax1.grid(True, alpha=0.3)

    # Graphique 2: Delta et nombre d'actions
    ax2 = axes[0, 1]
    ax2.plot(df['time'], df['delta'], 'g-', linewidth=1.5, label='Delta')
    ax2.set_xlabel('Temps (années)')
    ax2.set_ylabel('Delta / Actions détenues')
    ax2.set_title('Évolution du Delta')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Graphique 3: Valeur de l'option vs couverture
    ax3 = axes[1, 0]
    ax3.plot(df['time'], df['option_value'], 'r-', linewidth=1.5, label='Valeur option')
    ax3.set_xlabel('Temps (années)')
    ax3.set_ylabel('Valeur (€)')
    ax3.set_title('Valeur de l\'option')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Graphique 4: Erreur de couverture
    ax4 = axes[1, 1]
    ax4.plot(df['time'], df['portfolio_value'], 'm-', linewidth=1.5)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Temps (années)')
    ax4.set_ylabel('Valeur du portefeuille (€)')
    ax4.set_title(f'Erreur de couverture (final: {summary["hedging_error"]:.4f}€)')
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def run_multiple_simulations(S0, K, T, r, sigma, n_simulations=100, n_steps=252):
    """
    Exécute plusieurs simulations pour analyser la distribution des erreurs.

    Paramètres:
    -----------
    S0, K, T, r, sigma : paramètres BSM standard
    n_simulations : int
        Nombre de simulations
    n_steps : int
        Nombre de rebalancements par simulation

    Retourne:
    ---------
    dict : statistiques sur les erreurs de couverture
    """
    errors = []
    error_percents = []

    for i in range(n_simulations):
        _, summary = simulate_delta_hedging(S0, K, T, r, sigma, n_steps, seed=i)
        errors.append(summary['hedging_error'])
        error_percents.append(summary['error_percent'])

    errors = np.array(errors)
    error_percents = np.array(error_percents)

    return {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'mean_error_percent': np.mean(error_percents),
        'std_error_percent': np.std(error_percents),
        'max_error': np.max(np.abs(errors)),
        'errors': errors
    }


# Tests si exécuté directement
if __name__ == "__main__":
    # Paramètres de test
    S0 = 100    # Prix initial
    K = 100     # Strike ATM
    T = 1.0     # 1 an
    r = 0.05    # 5%
    sigma = 0.2 # 20%

    print("=== Test Delta Hedging ===\n")
    print(f"Paramètres: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")

    # Une simulation détaillée
    df, summary = simulate_delta_hedging(S0, K, T, r, sigma, n_steps=252, seed=42)

    print(f"\n=== Résultat d'une simulation ===")
    print(f"Prime de l'option vendue: {summary['option_premium']:.4f} €")
    print(f"Prix final du sous-jacent: {summary['final_stock_price']:.2f} €")
    print(f"Payoff de l'option: {summary['option_payoff']:.4f} €")
    print(f"Erreur de couverture: {summary['hedging_error']:.4f} €")
    print(f"Erreur en %: {summary['error_percent']:.2f}%")

    # Tracer les graphiques
    fig = plot_hedging_simulation(df, summary)
    plt.savefig('delta_hedging_simulation.png', dpi=150, bbox_inches='tight')
    print("\nGraphique sauvegardé: delta_hedging_simulation.png")

    # Analyse sur plusieurs simulations
    print("\n=== Analyse sur 100 simulations ===")
    stats = run_multiple_simulations(S0, K, T, r, sigma, n_simulations=100)
    print(f"Erreur moyenne: {stats['mean_error']:.4f} € (±{stats['std_error']:.4f})")
    print(f"Erreur moyenne en %: {stats['mean_error_percent']:.2f}% (±{stats['std_error_percent']:.2f}%)")
    print(f"Erreur maximale: {stats['max_error']:.4f} €")
