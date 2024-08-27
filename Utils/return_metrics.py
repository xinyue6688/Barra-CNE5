import numpy as np
import pandas as pd

class MetricsCalculator:
    def __init__(self, label, daily_return, date):
        self.label = label
        self.daily_return = daily_return
        self.date = date
        self.calculate_metrics()

    def calculate_metrics(self):
        self.df = pd.DataFrame({
            'daily_return': self.daily_return,
            'date': self.date
        })
        # NAV
        self.nav = self.df['daily_return'].transform(lambda x: (1 + x).cumprod())
        # Total return
        self.total_return = (self.nav.iloc[-1] / self.nav.iloc[1]) - 1

        # Annualized return
        self.annualized_return = ((1 + self.total_return) ** (252 / len(self.df))) - 1

        # Daily volatility
        self.daily_volatility = self.df['daily_return'].std()

        # Annualized volatility
        self.annualized_volatility = self.daily_volatility * np.sqrt(252)

        # Sharpe ratio
        self.sharpe_ratio = self.annualized_return / self.annualized_volatility

        # Max drawdown
        cumulative_returns = (1 + self.df['daily_return']).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        self.max_drawdown = drawdown.min()

        # Daily win rate
        self.df['daily_win_rate'] = np.where(self.df['daily_return'] > 0, 1, 0)
        self.daily_win_rate = self.df['daily_win_rate'].mean()

        # Max drawdown start and end dates
        max_dd_id = np.argmax(np.maximum.accumulate(self.df['daily_return']) - self.df['daily_return'])
        drawdown_end_id = np.argmax(self.df['daily_return'][:max_dd_id])
        self.drawdown_end_date = self.df['date'].iloc[drawdown_end_id]
        drawdown_start_id = np.argmax(self.df['daily_return'][:drawdown_end_id])
        self.drawdown_start_date = self.df['date'].iloc[drawdown_start_id]

    def print_metrics(self):
        print(f"{self.label} Metrics:")
        print(f"Total Return: {self.total_return:.2%}")
        print(f"Annualized Return: {self.annualized_return:.2%}")
        print(f"Annualized Volatility: {self.annualized_volatility:.2%}")
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {self.max_drawdown:.2%}")
        print(f"Daily Win Rate: {self.daily_win_rate:.2%}")
        print(f"Max Drawdown Start Date: {self.drawdown_start_date}")
        print(f"Max Drawdown End Date: {self.drawdown_end_date}")


