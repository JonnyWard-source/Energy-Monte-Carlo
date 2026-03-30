import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

def download_data(ticker: str, start: str, end: str) -> pd.Series:
    """
    Dowmload historical data from Yahoo Finance.
    Parameters:
        ticker (str): Asset ticker (eg "CL=F" for crude oil)
        start (str): Start date (YYYY-MM-DD)
        end (str): End date (YYYY-MM-DD)

    Returns:
        pd.Series: Closing price series
    """
    data = yf.download(ticker, start, end)
    prices = data['Close'].dropna()
    return prices

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log returns from price data.
    """
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        returns = np.log(prices/prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return returns.dropna()

def estimate_parameters(returns: pd.Series) -> tuple:
    """
    Estimate drift (mu) and volatility (sigma).
    """
    mu = returns.mean()
    sigma = returns.std()
    return mu, sigma

def monte_carlo_simulation(S0: float, mu: float, sigma: float,
                           T: int = 252, N: int = 1000) -> np.ndarray:
    """
    Simulate price paths using Geometric Brownian Motion.

    Parameters:
        S0 (float): Initial price
        mu (float): mean return
        sigma (float): volatility
        T (int): Time horizon (days)
        N (int): Number of simulations

    Returns:
        np.ndarray: Simulated price paths (T x N)
    """
    dt = 1/252
    simulations = np.zeros((T, N))
    for i in range(N):
        price_path = [S0]
        for j in range(1, T):
            Z = np.random.normal()
            St = price_path[-1] * np.exp((mu-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
            price_path.append(St)
        simulations[:, i] = np.array(price_path).reshape(-1)
    return simulations

def analyse_simulation(simulations: np.ndarray) -> dict:
    """
    Compute summary statistics from simulations
    """
    final_prices = simulations[-1]
    return {
        "Expected_price": np.mean(final_prices),
        "5_percentile": np.percentile(final_prices, 5),
        "95_percentile": np.percentile(final_prices, 95)
    }

def plot_results(prices: pd.Series, simulations: np.ndarray) -> None:
    """
    Plot historical prices and simulated paths
    """
    plt.figure(figsize=(10, 5))
    plt.plot(prices)
    plt.title("Historical prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(simulations[:, :50])
    plt.title("Monte Carlo Simulated paths")
    plt.xlabel("Time (days)")
    plt.ylabel("Price")
    plt.show()

def main():
    # Parameters
    ticker = "CL=F"
    start = "2020-01-01"
    end = ("2024-01-01")

    # Pipeline
    prices = download_data(ticker, start, end)
    returns = compute_log_returns(prices)
    mu, sigma = estimate_parameters(returns)
    simulations = monte_carlo_simulation(prices.iloc[-1], mu, sigma)
    results = analyse_simulation(simulations)

    # Output
    print(f"Mean return: {mu}")
    print(f"Volatility: {sigma}")
    print("\nSimulation Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

    # Plot
    plot_results(prices, simulations)

if __name__ == '__main__':
    main()
