# -*- coding: utf-8 -*-
"""
Analyse du Volatility Smile
===========================
Visualisation et analyse de la structure de volatilité implicite
en fonction du strike (moneyness).

Référence: Hull, Chapter 20 - Volatility Smiles

Le "smile" de volatilité est le phénomène où la volatilité implicite
varie selon le strike, formant souvent une courbe en "U" ou asymétrique.

Observations typiques:
    - Options equity: "smirk" (vol plus élevée pour les puts OTM)
    - Options FX: smile symétrique
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .implied_volatility import calculate_implied_vol


def calculate_moneyness(S, K):
    """
    Calcule le moneyness d'une option.

    Le moneyness mesure à quel point l'option est ITM ou OTM.

    Définition utilisée ici: K/S
        - K/S < 1 : Call ITM / Put OTM
        - K/S = 1 : ATM (At The Money)
        - K/S > 1 : Call OTM / Put ITM

    Paramètres:
    -----------
    S : float
        Prix spot
    K : float
        Strike

    Retourne:
    ---------
    float : moneyness (K/S)
    """
    return K / S


def analyze_volatility_smile(option_data, S, T, r):
    """
    Analyse le volatility smile à partir de données d'options.

    Paramètres:
    -----------
    option_data : list of dict
        Liste de dictionnaires avec 'strike', 'price', 'type' (call/put)
    S : float
        Prix spot actuel
    T : float
        Temps jusqu'à maturité
    r : float
        Taux sans risque

    Retourne:
    ---------
    pd.DataFrame : tableau avec strike, moneyness, vol implicite

    Exemple:
    --------
    >>> data = [
    ...     {'strike': 90, 'price': 15.5, 'type': 'call'},
    ...     {'strike': 100, 'price': 10.5, 'type': 'call'},
    ...     {'strike': 110, 'price': 6.8, 'type': 'call'}
    ... ]
    >>> df = analyze_volatility_smile(data, S=100, T=1, r=0.05)
    """
    results = []

    for option in option_data:
        K = option['strike']
        price = option['price']
        opt_type = option['type']

        # Calcul de la vol implicite
        iv = calculate_implied_vol(price, S, K, T, r, opt_type)

        # Calcul du moneyness
        moneyness = calculate_moneyness(S, K)

        results.append({
            'strike': K,
            'moneyness': moneyness,
            'price': price,
            'type': opt_type,
            'implied_vol': iv
        })

    # Créer un DataFrame et trier par strike
    df = pd.DataFrame(results)
    df = df.sort_values('strike')

    return df


def plot_volatility_smile(df, title="Volatility Smile"):
    """
    Trace le graphique du volatility smile.

    Paramètres:
    -----------
    df : pd.DataFrame
        DataFrame avec colonnes 'strike' ou 'moneyness' et 'implied_vol'
    title : str
        Titre du graphique

    Retourne:
    ---------
    matplotlib.figure.Figure : figure du graphique
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Graphique 1: Vol implicite vs Strike
    ax1 = axes[0]
    ax1.plot(df['strike'], df['implied_vol'] * 100, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Strike (K)', fontsize=12)
    ax1.set_ylabel('Volatilité Implicite (%)', fontsize=12)
    ax1.set_title('Volatilité vs Strike', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=df['implied_vol'].mean() * 100, color='r', linestyle='--',
                label=f'Moyenne: {df["implied_vol"].mean()*100:.1f}%')
    ax1.legend()

    # Graphique 2: Vol implicite vs Moneyness
    ax2 = axes[1]
    ax2.plot(df['moneyness'], df['implied_vol'] * 100, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Moneyness (K/S)', fontsize=12)
    ax2.set_ylabel('Volatilité Implicite (%)', fontsize=12)
    ax2.set_title('Volatilité vs Moneyness', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=1.0, color='r', linestyle='--', label='ATM (K/S=1)')
    ax2.legend()

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


def calculate_smile_metrics(df):
    """
    Calcule les métriques caractéristiques du smile.

    Paramètres:
    -----------
    df : pd.DataFrame
        DataFrame du volatility smile

    Retourne:
    ---------
    dict : métriques du smile

    Métriques calculées:
        - atm_vol: volatilité ATM (moneyness ~ 1)
        - skew: différence entre vol OTM put et ATM
        - smile_curvature: convexité du smile
    """
    # Volatilité ATM (la plus proche de moneyness = 1)
    atm_idx = (df['moneyness'] - 1.0).abs().idxmin()
    atm_vol = df.loc[atm_idx, 'implied_vol']

    # Volatilité des puts OTM (moneyness > 1.1)
    otm_puts = df[df['moneyness'] > 1.10]['implied_vol']
    otm_put_vol = otm_puts.mean() if len(otm_puts) > 0 else np.nan

    # Volatilité des calls OTM (moneyness < 0.9)
    otm_calls = df[df['moneyness'] < 0.90]['implied_vol']
    otm_call_vol = otm_calls.mean() if len(otm_calls) > 0 else np.nan

    # Skew: différence entre vol des puts OTM et ATM
    skew = otm_put_vol - atm_vol if not np.isnan(otm_put_vol) else np.nan

    # Smile (symétrie): moyenne des vols extrêmes vs ATM
    if not np.isnan(otm_put_vol) and not np.isnan(otm_call_vol):
        smile_curvature = ((otm_put_vol + otm_call_vol) / 2) - atm_vol
    else:
        smile_curvature = np.nan

    return {
        'atm_vol': atm_vol,
        'otm_put_vol': otm_put_vol,
        'otm_call_vol': otm_call_vol,
        'skew': skew,
        'smile_curvature': smile_curvature,
        'vol_range': df['implied_vol'].max() - df['implied_vol'].min()
    }


def generate_synthetic_smile_data(S, T, r, base_vol=0.2, skew=-0.1, convexity=0.05,
                                    strikes=None):
    """
    Génère des données synthétiques de volatility smile pour les tests.

    Le modèle utilisé est quadratique:
        σ(K) = σ_ATM + skew * (K/S - 1) + convexity * (K/S - 1)²

    Paramètres:
    -----------
    S : float
        Prix spot
    T : float
        Maturité
    r : float
        Taux sans risque
    base_vol : float
        Volatilité ATM de base
    skew : float
        Pente du smile (négatif = smirk typique actions)
    convexity : float
        Convexité du smile
    strikes : list
        Liste des strikes (défaut: 80% à 120% du spot)

    Retourne:
    ---------
    list : données d'options pour analyze_volatility_smile
    """
    from .black_scholes import black_scholes_price

    if strikes is None:
        # Strikes de 80% à 120% du spot, par pas de 5%
        strikes = [S * (0.80 + i * 0.05) for i in range(9)]

    option_data = []

    for K in strikes:
        # Moneyness
        m = K / S

        # Volatilité selon le modèle quadratique
        sigma = base_vol + skew * (m - 1) + convexity * (m - 1) ** 2

        # S'assurer que la vol est positive
        sigma = max(sigma, 0.05)

        # Calculer le prix avec cette volatilité
        opt_type = 'call' if K >= S else 'put'
        price = black_scholes_price(S, K, T, r, sigma, opt_type)

        option_data.append({
            'strike': K,
            'price': price,
            'type': opt_type,
            'true_vol': sigma  # Pour vérification
        })

    return option_data


# Tests si exécuté directement
if __name__ == "__main__":
    # Paramètres de test
    S = 100  # Prix spot
    T = 0.5  # 6 mois
    r = 0.05 # 5%

    print("=== Test Volatility Smile ===\n")

    # Générer des données synthétiques avec un smirk typique
    option_data = generate_synthetic_smile_data(
        S=S, T=T, r=r,
        base_vol=0.20,   # Vol ATM = 20%
        skew=-0.15,      # Smirk négatif (typique actions)
        convexity=0.10   # Convexité positive
    )

    # Analyser le smile
    df = analyze_volatility_smile(option_data, S, T, r)

    print("Données du Volatility Smile:")
    print(df.to_string(index=False))

    # Calculer les métriques
    metrics = calculate_smile_metrics(df)
    print(f"\n=== Métriques du Smile ===")
    print(f"Vol ATM: {metrics['atm_vol']:.2%}")
    print(f"Vol OTM Put: {metrics['otm_put_vol']:.2%}" if not np.isnan(metrics['otm_put_vol']) else "Vol OTM Put: N/A")
    print(f"Vol OTM Call: {metrics['otm_call_vol']:.2%}" if not np.isnan(metrics['otm_call_vol']) else "Vol OTM Call: N/A")
    print(f"Skew: {metrics['skew']:.2%}" if not np.isnan(metrics['skew']) else "Skew: N/A")
    print(f"Range de vol: {metrics['vol_range']:.2%}")

    # Tracer le graphique
    fig = plot_volatility_smile(df, title="Volatility Smile - Données Synthétiques")
    plt.savefig('volatility_smile.png', dpi=150, bbox_inches='tight')
    print("\nGraphique sauvegardé: volatility_smile.png")
