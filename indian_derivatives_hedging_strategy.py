"""
Indian Derivatives Market One-Week Hedging Strategy
Professional Quantitative Analysis for Options & Futures
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy.special import erf
import warnings
warnings.filterwarnings('ignore')

# Market Parameters (as of current market conditions)
NIFTY_SPOT = 24500
BANKNIFTY_SPOT = 52000
INDIA_VIX = 14.5
RISK_FREE_RATE = 0.065  # 6.5% annual
TRADING_DAYS = 252
TRANSACTION_COST_PERCENTAGE = 0.0005  # 0.05% per side
STT_OPTIONS = 0.125  # Securities Transaction Tax for options
STT_FUTURES = 0.0125  # STT for futures

@dataclass
class OptionContract:
    """Option contract specifications"""
    underlying: str
    strike: float
    option_type: str  # 'CE' or 'PE'
    expiry: datetime
    lot_size: int
    premium: float
    implied_volatility: float
    
@dataclass
class FuturesContract:
    """Futures contract specifications"""
    underlying: str
    expiry: datetime
    lot_size: int
    price: float
    
@dataclass
class GreeksCalculation:
    """Option Greeks"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

class BlackScholesModel:
    """Black-Scholes model for option pricing and Greeks calculation"""
    
    @staticmethod
    def normal_cdf(x):
        """Cumulative distribution function for standard normal"""
        return 0.5 * (1 + erf(x / np.sqrt(2)))
    
    @staticmethod
    def normal_pdf(x):
        """Probability density function for standard normal"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    @staticmethod
    def calculate_d1_d2(S, K, r, T, sigma):
        """Calculate d1 and d2 for Black-Scholes formula"""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return d1, d2
    
    @classmethod
    def calculate_option_price(cls, S, K, r, T, sigma, option_type='CE'):
        """Calculate option price using Black-Scholes formula"""
        d1, d2 = cls.calculate_d1_d2(S, K, r, T, sigma)
        
        if option_type == 'CE':
            price = S * cls.normal_cdf(d1) - K * np.exp(-r*T) * cls.normal_cdf(d2)
        else:  # PE
            price = K * np.exp(-r*T) * cls.normal_cdf(-d2) - S * cls.normal_cdf(-d1)
        
        return price
    
    @classmethod
    def calculate_greeks(cls, S, K, r, T, sigma, option_type='CE') -> GreeksCalculation:
        """Calculate all Greeks for an option"""
        d1, d2 = cls.calculate_d1_d2(S, K, r, T, sigma)
        
        # Delta
        if option_type == 'CE':
            delta = cls.normal_cdf(d1)
        else:
            delta = cls.normal_cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = cls.normal_pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        if option_type == 'CE':
            theta = (-S * cls.normal_pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r*T) * cls.normal_cdf(d2)) / 365
        else:
            theta = (-S * cls.normal_pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r*T) * cls.normal_cdf(-d2)) / 365
        
        # Vega (same for calls and puts)
        vega = S * cls.normal_pdf(d1) * np.sqrt(T) / 100
        
        # Rho
        if option_type == 'CE':
            rho = K * T * np.exp(-r*T) * cls.normal_cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r*T) * cls.normal_cdf(-d2) / 100
        
        return GreeksCalculation(delta, gamma, theta, vega, rho)

class PortfolioHedgingStrategy:
    """Comprehensive one-week hedging strategy for Indian derivatives"""
    
    def __init__(self, portfolio_value: float, portfolio_beta: float = 1.0):
        self.portfolio_value = portfolio_value
        self.portfolio_beta = portfolio_beta
        self.current_date = datetime.now()
        self.expiry_date = self.get_weekly_expiry()
        self.time_to_expiry = (self.expiry_date - self.current_date).days / 365
        
    def get_weekly_expiry(self) -> datetime:
        """Get next Thursday expiry for weekly options"""
        days_ahead = 3 - self.current_date.weekday()  # Thursday is 3
        if days_ahead <= 0:
            days_ahead += 7
        return self.current_date + timedelta(days=days_ahead)
    
    def analyze_market_conditions(self) -> Dict:
        """Analyze current market conditions"""
        analysis = {
            'nifty_level': NIFTY_SPOT,
            'banknifty_level': BANKNIFTY_SPOT,
            'vix_level': INDIA_VIX,
            'vix_percentile': self.calculate_vix_percentile(),
            'market_regime': self.determine_market_regime(),
            'expected_weekly_move': self.calculate_expected_move()
        }
        return analysis
    
    def calculate_vix_percentile(self) -> float:
        """Calculate VIX percentile based on historical data"""
        # Simplified: assuming current VIX is at 40th percentile
        return 40.0
    
    def determine_market_regime(self) -> str:
        """Determine current market regime"""
        if INDIA_VIX < 12:
            return "Low Volatility"
        elif INDIA_VIX < 18:
            return "Normal Volatility"
        elif INDIA_VIX < 25:
            return "Elevated Volatility"
        else:
            return "High Volatility"
    
    def calculate_expected_move(self) -> Dict[str, float]:
        """Calculate expected weekly move based on VIX"""
        weekly_vol = INDIA_VIX / np.sqrt(52)
        nifty_move = NIFTY_SPOT * weekly_vol / 100
        banknifty_move = BANKNIFTY_SPOT * weekly_vol / 100
        
        return {
            'nifty_expected_move': nifty_move,
            'banknifty_expected_move': banknifty_move,
            'nifty_upper': NIFTY_SPOT + nifty_move,
            'nifty_lower': NIFTY_SPOT - nifty_move,
            'banknifty_upper': BANKNIFTY_SPOT + banknifty_move,
            'banknifty_lower': BANKNIFTY_SPOT - banknifty_move
        }
    
    def select_hedging_instruments(self) -> Dict:
        """Select appropriate hedging instruments"""
        market_analysis = self.analyze_market_conditions()
        expected_moves = self.calculate_expected_move()
        
        instruments = {
            'primary_hedge': self.select_primary_hedge(expected_moves),
            'volatility_hedge': self.select_volatility_hedge(),
            'tail_risk_hedge': self.select_tail_risk_hedge(expected_moves),
            'income_generation': self.select_income_strategy()
        }
        
        return instruments
    
    def select_primary_hedge(self, expected_moves: Dict) -> List[OptionContract]:
        """Select primary hedging instruments"""
        # Put spread for downside protection
        nifty_atm_strike = round(NIFTY_SPOT / 50) * 50
        put_long_strike = nifty_atm_strike - 100
        put_short_strike = nifty_atm_strike - 300
        
        contracts = [
            OptionContract(
                underlying='NIFTY',
                strike=put_long_strike,
                option_type='PE',
                expiry=self.expiry_date,
                lot_size=50,
                premium=self.calculate_option_premium(NIFTY_SPOT, put_long_strike, 'PE'),
                implied_volatility=INDIA_VIX * 1.1
            ),
            OptionContract(
                underlying='NIFTY',
                strike=put_short_strike,
                option_type='PE',
                expiry=self.expiry_date,
                lot_size=50,
                premium=self.calculate_option_premium(NIFTY_SPOT, put_short_strike, 'PE'),
                implied_volatility=INDIA_VIX * 1.2
            )
        ]
        
        return contracts
    
    def select_volatility_hedge(self) -> List[OptionContract]:
        """Select volatility hedging strategy"""
        # Iron Condor for range-bound market
        nifty_atm_strike = round(NIFTY_SPOT / 50) * 50
        
        contracts = [
            # Bull Put Spread
            OptionContract('NIFTY', nifty_atm_strike - 200, 'PE', self.expiry_date, 50,
                         self.calculate_option_premium(NIFTY_SPOT, nifty_atm_strike - 200, 'PE'),
                         INDIA_VIX * 1.15),
            OptionContract('NIFTY', nifty_atm_strike - 300, 'PE', self.expiry_date, 50,
                         self.calculate_option_premium(NIFTY_SPOT, nifty_atm_strike - 300, 'PE'),
                         INDIA_VIX * 1.2),
            # Bear Call Spread
            OptionContract('NIFTY', nifty_atm_strike + 200, 'CE', self.expiry_date, 50,
                         self.calculate_option_premium(NIFTY_SPOT, nifty_atm_strike + 200, 'CE'),
                         INDIA_VIX * 1.15),
            OptionContract('NIFTY', nifty_atm_strike + 300, 'CE', self.expiry_date, 50,
                         self.calculate_option_premium(NIFTY_SPOT, nifty_atm_strike + 300, 'CE'),
                         INDIA_VIX * 1.2)
        ]
        
        return contracts
    
    def select_tail_risk_hedge(self, expected_moves: Dict) -> List[OptionContract]:
        """Select tail risk hedging instruments"""
        # Far OTM puts for black swan protection
        nifty_5_percent_down = NIFTY_SPOT * 0.95
        strike = round(nifty_5_percent_down / 100) * 100
        
        return [
            OptionContract(
                underlying='NIFTY',
                strike=strike,
                option_type='PE',
                expiry=self.expiry_date,
                lot_size=50,
                premium=self.calculate_option_premium(NIFTY_SPOT, strike, 'PE'),
                implied_volatility=INDIA_VIX * 1.5
            )
        ]
    
    def select_income_strategy(self) -> List[OptionContract]:
        """Select income generation strategy"""
        # Covered call on Bank Nifty
        banknifty_otm_strike = round(BANKNIFTY_SPOT * 1.015 / 100) * 100
        
        return [
            OptionContract(
                underlying='BANKNIFTY',
                strike=banknifty_otm_strike,
                option_type='CE',
                expiry=self.expiry_date,
                lot_size=25,
                premium=self.calculate_option_premium(BANKNIFTY_SPOT, banknifty_otm_strike, 'CE'),
                implied_volatility=INDIA_VIX * 1.3
            )
        ]
    
    def calculate_option_premium(self, spot: float, strike: float, option_type: str) -> float:
        """Calculate option premium using Black-Scholes"""
        sigma = INDIA_VIX / 100
        premium = BlackScholesModel.calculate_option_price(
            S=spot,
            K=strike,
            r=RISK_FREE_RATE,
            T=self.time_to_expiry,
            sigma=sigma,
            option_type=option_type
        )
        return round(premium, 2)
    
    def calculate_position_sizes(self, instruments: Dict) -> Dict:
        """Calculate optimal position sizes based on portfolio value and risk parameters"""
        hedge_ratio = self.portfolio_beta
        portfolio_delta_exposure = self.portfolio_value * hedge_ratio
        
        position_sizes = {}
        
        # Primary hedge sizing (40% of risk budget)
        primary_contracts = instruments['primary_hedge']
        if primary_contracts:
            put_spread_delta = -0.35  # Approximate delta of put spread
            contracts_needed = abs(portfolio_delta_exposure * 0.4 / 
                                 (NIFTY_SPOT * 50 * put_spread_delta))
            position_sizes['primary_hedge'] = round(contracts_needed)
        
        # Volatility hedge sizing (20% of risk budget)
        position_sizes['volatility_hedge'] = round(self.portfolio_value * 0.002 / 
                                                  (NIFTY_SPOT * 50))
        
        # Tail risk hedge sizing (10% of risk budget)
        position_sizes['tail_risk_hedge'] = round(self.portfolio_value * 0.001 / 
                                                (NIFTY_SPOT * 50))
        
        # Income generation sizing (30% of risk budget)
        position_sizes['income_generation'] = round(self.portfolio_value * 0.003 / 
                                                  (BANKNIFTY_SPOT * 25))
        
        return position_sizes
    
    def create_risk_assessment_matrix(self, instruments: Dict, position_sizes: Dict) -> pd.DataFrame:
        """Create comprehensive risk assessment matrix"""
        scenarios = [
            {'name': 'Base Case', 'nifty_move': 0, 'vix_move': 0},
            {'name': 'Mild Bullish', 'nifty_move': 1, 'vix_move': -10},
            {'name': 'Strong Bullish', 'nifty_move': 3, 'vix_move': -20},
            {'name': 'Mild Bearish', 'nifty_move': -1, 'vix_move': 10},
            {'name': 'Strong Bearish', 'nifty_move': -3, 'vix_move': 30},
            {'name': 'Black Swan', 'nifty_move': -5, 'vix_move': 100}
        ]
        
        risk_matrix = []
        
        for scenario in scenarios:
            nifty_new = NIFTY_SPOT * (1 + scenario['nifty_move']/100)
            vix_new = INDIA_VIX * (1 + scenario['vix_move']/100)
            
            portfolio_pnl = self.portfolio_value * scenario['nifty_move']/100 * self.portfolio_beta
            hedge_pnl = self.calculate_hedge_pnl(instruments, position_sizes, nifty_new, vix_new)
            net_pnl = portfolio_pnl + hedge_pnl
            
            risk_matrix.append({
                'Scenario': scenario['name'],
                'Nifty_Level': nifty_new,
                'VIX_Level': vix_new,
                'Portfolio_PnL': portfolio_pnl,
                'Hedge_PnL': hedge_pnl,
                'Net_PnL': net_pnl,
                'Net_Return_%': (net_pnl / self.portfolio_value) * 100
            })
        
        return pd.DataFrame(risk_matrix)
    
    def calculate_hedge_pnl(self, instruments: Dict, position_sizes: Dict, 
                          nifty_new: float, vix_new: float) -> float:
        """Calculate hedge P&L for given market scenario"""
        total_pnl = 0
        
        # Simplified P&L calculation
        # Primary hedge (put spread)
        if 'primary_hedge' in instruments and instruments['primary_hedge']:
            contracts = instruments['primary_hedge']
            size = position_sizes.get('primary_hedge', 0)
            
            # Long put P&L
            long_put = contracts[0]
            long_put_value = BlackScholesModel.calculate_option_price(
                S=nifty_new, K=long_put.strike, r=RISK_FREE_RATE,
                T=self.time_to_expiry * 0.8,  # Assuming 20% time decay
                sigma=vix_new/100, option_type='PE'
            )
            long_put_pnl = (long_put_value - long_put.premium) * 50 * size
            
            # Short put P&L
            if len(contracts) > 1:
                short_put = contracts[1]
                short_put_value = BlackScholesModel.calculate_option_price(
                    S=nifty_new, K=short_put.strike, r=RISK_FREE_RATE,
                    T=self.time_to_expiry * 0.8,
                    sigma=vix_new/100, option_type='PE'
                )
                short_put_pnl = (short_put.premium - short_put_value) * 50 * size
                total_pnl += long_put_pnl + short_put_pnl
        
        # Add other strategy P&L calculations...
        # Simplified for brevity
        
        return total_pnl
    
    def generate_entry_exit_criteria(self) -> Dict:
        """Generate specific entry and exit criteria"""
        criteria = {
            'entry_conditions': {
                'vix_threshold': INDIA_VIX * 0.9,
                'time_window': 'First 30 minutes after market open',
                'price_levels': {
                    'nifty_range': (NIFTY_SPOT * 0.998, NIFTY_SPOT * 1.002),
                    'banknifty_range': (BANKNIFTY_SPOT * 0.997, BANKNIFTY_SPOT * 1.003)
                },
                'execution_style': 'Scaled entry over 15-minute window'
            },
            'exit_conditions': {
                'profit_target': self.portfolio_value * 0.015,  # 1.5% profit
                'stop_loss': self.portfolio_value * 0.02,  # 2% loss
                'time_exit': 'Thursday 3:15 PM (15 minutes before expiry)',
                'volatility_spike': INDIA_VIX * 1.5,
                'delta_threshold': 0.7  # Exit if position delta exceeds threshold
            },
            'adjustment_triggers': {
                'delta_breach': 0.5,
                'vega_breach': self.portfolio_value * 0.001,
                'profit_lock': 0.01  # Lock 1% profit
            }
        }
        
        return criteria
    
    def calculate_transaction_costs(self, instruments: Dict, position_sizes: Dict) -> float:
        """Calculate total transaction costs including STT, brokerage, and charges"""
        total_cost = 0
        
        for strategy, contracts in instruments.items():
            if isinstance(contracts, list) and contracts:
                size = position_sizes.get(strategy, 0)
                for contract in contracts:
                    # Brokerage (flat rate assumed)
                    brokerage = 20 * size * 2  # Entry and exit
                    
                    # STT on premium
                    stt = contract.premium * size * contract.lot_size * STT_OPTIONS / 100
                    
                    # Exchange charges and GST
                    exchange_charges = contract.premium * size * contract.lot_size * 0.00053
                    gst = (brokerage + exchange_charges) * 0.18
                    
                    total_cost += brokerage + stt + exchange_charges + gst
        
        return total_cost
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate complete hedging strategy report"""
        # Analyze market
        market_analysis = self.analyze_market_conditions()
        
        # Select instruments
        instruments = self.select_hedging_instruments()
        
        # Calculate positions
        position_sizes = self.calculate_position_sizes(instruments)
        
        # Risk assessment
        risk_matrix = self.create_risk_assessment_matrix(instruments, position_sizes)
        
        # Entry/exit criteria
        entry_exit = self.generate_entry_exit_criteria()
        
        # Transaction costs
        transaction_costs = self.calculate_transaction_costs(instruments, position_sizes)
        
        # Calculate key metrics
        max_profit = risk_matrix['Net_PnL'].max()
        max_loss = risk_matrix['Net_PnL'].min()
        expected_return = risk_matrix['Net_PnL'].mean()
        
        report = {
            'execution_date': self.current_date.strftime('%Y-%m-%d'),
            'expiry_date': self.expiry_date.strftime('%Y-%m-%d'),
            'portfolio_value': self.portfolio_value,
            'portfolio_beta': self.portfolio_beta,
            'market_analysis': market_analysis,
            'hedging_instruments': self.format_instruments(instruments),
            'position_sizes': position_sizes,
            'risk_assessment': risk_matrix.to_dict('records'),
            'entry_exit_criteria': entry_exit,
            'transaction_costs': transaction_costs,
            'key_metrics': {
                'max_profit': max_profit,
                'max_loss': max_loss,
                'expected_return': expected_return,
                'cost_as_percentage': (transaction_costs / self.portfolio_value) * 100,
                'break_even_move': abs(transaction_costs / (self.portfolio_value * self.portfolio_beta)) * 100
            },
            'recommendations': self.generate_recommendations()
        }
        
        return report
    
    def format_instruments(self, instruments: Dict) -> Dict:
        """Format instruments for report"""
        formatted = {}
        for strategy, contracts in instruments.items():
            if isinstance(contracts, list):
                formatted[strategy] = []
                for contract in contracts:
                    formatted[strategy].append({
                        'underlying': contract.underlying,
                        'strike': contract.strike,
                        'type': contract.option_type,
                        'premium': contract.premium,
                        'lot_size': contract.lot_size
                    })
        return formatted
    
    def generate_recommendations(self) -> List[str]:
        """Generate strategic recommendations"""
        recommendations = [
            "Execute primary hedge immediately at market open for optimal liquidity",
            "Monitor VIX levels closely - adjust position if VIX moves beyond 12-18 range",
            "Consider rolling positions on Wednesday if time decay exceeds 50% of premium",
            "Maintain stop-loss discipline - exit if portfolio loss exceeds 2%",
            "Book partial profits at 1% portfolio gain to reduce risk",
            "Keep 20% capital as reserve for intraday adjustments",
            "Track overnight global markets for gap risk assessment",
            "Use limit orders for entry to minimize slippage costs"
        ]
        
        if INDIA_VIX > 15:
            recommendations.append("High VIX environment - consider reducing position sizes by 20%")
        
        return recommendations

# Example execution
def main():
    """Execute hedging strategy analysis"""
    # Example portfolio: ₹1 Crore with beta of 1.2
    portfolio_value = 10000000  # ₹1 Crore
    portfolio_beta = 1.2
    
    # Initialize strategy
    hedging_strategy = PortfolioHedgingStrategy(portfolio_value, portfolio_beta)
    
    # Generate comprehensive report
    report = hedging_strategy.generate_comprehensive_report()
    
    # Display key results
    print("=== INDIAN DERIVATIVES ONE-WEEK HEDGING STRATEGY ===")
    print(f"\nPortfolio Value: ₹{portfolio_value:,.0f}")
    print(f"Portfolio Beta: {portfolio_beta}")
    print(f"\nMarket Conditions:")
    print(f"- Nifty: {report['market_analysis']['nifty_level']}")
    print(f"- India VIX: {report['market_analysis']['vix_level']}")
    print(f"- Market Regime: {report['market_analysis']['market_regime']}")
    
    print(f"\nExpected Weekly Move:")
    expected_moves = report['market_analysis']['expected_weekly_move']
    print(f"- Nifty: ±{expected_moves['nifty_expected_move']:.0f} points")
    print(f"- Bank Nifty: ±{expected_moves['banknifty_expected_move']:.0f} points")
    
    print(f"\nPosition Sizes:")
    for strategy, size in report['position_sizes'].items():
        print(f"- {strategy.replace('_', ' ').title()}: {size} contracts")
    
    print(f"\nRisk Metrics:")
    print(f"- Maximum Profit: ₹{report['key_metrics']['max_profit']:,.0f}")
    print(f"- Maximum Loss: ₹{report['key_metrics']['max_loss']:,.0f}")
    print(f"- Expected Return: ₹{report['key_metrics']['expected_return']:,.0f}")
    print(f"- Transaction Costs: ₹{report['transaction_costs']:,.0f} ({report['key_metrics']['cost_as_percentage']:.2f}%)")
    
    print(f"\nRisk Assessment by Scenario:")
    risk_df = pd.DataFrame(report['risk_assessment'])
    print(risk_df[['Scenario', 'Nifty_Level', 'Net_PnL', 'Net_Return_%']].to_string(index=False))
    
    print(f"\nKey Recommendations:")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        print(f"{i}. {rec}")
    
    # Export detailed report
    detailed_report = pd.DataFrame([report])
    detailed_report.to_csv('hedging_strategy_report.csv', index=False)
    
    # Export risk matrix
    risk_df.to_csv('risk_assessment_matrix.csv', index=False)
    
    return report

if __name__ == "__main__":
    report = main() 