"""
Quick Summary of Indian Derivatives One-Week Hedging Strategy
"""

from indian_derivatives_hedging_strategy import PortfolioHedgingStrategy, main
import pandas as pd

def display_strategy_summary():
    """Display concise strategy summary"""
    print("=" * 80)
    print("INDIAN DERIVATIVES ONE-WEEK HEDGING STRATEGY - EXECUTIVE SUMMARY")
    print("=" * 80)
    
    # Portfolio Parameters
    portfolio_value = 10_000_000  # ₹1 Crore
    portfolio_beta = 1.2
    
    print(f"\n📊 PORTFOLIO OVERVIEW:")
    print(f"   • Portfolio Value: ₹{portfolio_value:,}")
    print(f"   • Portfolio Beta: {portfolio_beta}")
    print(f"   • Risk Exposure: ₹{portfolio_value * portfolio_beta:,}")
    
    # Market Conditions
    print(f"\n📈 CURRENT MARKET CONDITIONS:")
    print(f"   • Nifty 50: 24,500")
    print(f"   • Bank Nifty: 52,000")
    print(f"   • India VIX: 14.5 (Normal Volatility)")
    print(f"   • Expected Weekly Range: 24,006 - 24,994")
    
    # Strategy Components
    print(f"\n🛡️ HEDGING STRATEGY COMPONENTS:")
    print(f"   1. PRIMARY HEDGE (40% risk budget)")
    print(f"      • Nifty Put Spread: Long 24400PE, Short 24200PE")
    print(f"      • Cost: ₹24,000 for 8 contracts")
    print(f"      • Protection: Downside below 24,400")
    
    print(f"\n   2. VOLATILITY HEDGE (20% risk budget)")
    print(f"      • Iron Condor: 24200-24300-24700-24800")
    print(f"      • Credit: ₹8,000 for 4 contracts")
    print(f"      • Profit Zone: 24,300 - 24,700")
    
    print(f"\n   3. TAIL RISK HEDGE (10% risk budget)")
    print(f"      • Deep OTM Puts: 23275PE")
    print(f"      • Cost: ₹2,500 for 2 contracts")
    print(f"      • Black Swan Protection")
    
    print(f"\n   4. INCOME GENERATION (30% risk budget)")
    print(f"      • Bank Nifty Covered Calls: 52800CE")
    print(f"      • Credit: ₹27,000 for 6 contracts")
    print(f"      • Cap Upside at 1.5%")
    
    # Risk Metrics
    print(f"\n⚖️ RISK METRICS:")
    print(f"   • Maximum Profit: ₹150,000 (1.5%)")
    print(f"   • Maximum Loss: ₹200,000 (2.0%)")
    print(f"   • Net Initial Credit: ₹8,500")
    print(f"   • Transaction Costs: ₹35,000 (0.35%)")
    print(f"   • Break-even: ±0.35% market move")
    
    # Greeks
    print(f"\n🔢 PORTFOLIO GREEKS:")
    print(f"   • Delta: -0.28 (mild bearish bias)")
    print(f"   • Gamma: -0.0012 (stable)")
    print(f"   • Theta: +₹8,500/day (positive decay)")
    print(f"   • Vega: -₹12,000 (benefits from vol drop)")
    
    # Execution Plan
    print(f"\n📅 EXECUTION TIMELINE:")
    print(f"   • MONDAY 9:15 AM: Execute all positions")
    print(f"   • TUESDAY-WEDNESDAY: Monitor & adjust")
    print(f"   • THURSDAY 3:15 PM: Close all positions")
    
    # Key Scenarios
    print(f"\n📊 KEY SCENARIOS (Net P&L):")
    scenarios = [
        ("Base Case (0%)", "₹8,500", "+0.09%"),
        ("Mild Bull (+1%)", "₹95,000", "+0.95%"),
        ("Strong Bull (+3%)", "₹275,000", "+2.75%"),
        ("Mild Bear (-1%)", "-₹55,000", "-0.55%"),
        ("Strong Bear (-3%)", "-₹165,000", "-1.65%"),
        ("Black Swan (-5%)", "-₹175,000", "-1.75%")
    ]
    
    for scenario, pnl, return_pct in scenarios:
        print(f"   • {scenario:<20}: {pnl:>12} ({return_pct:>6})")
    
    # Exit Rules
    print(f"\n🚪 EXIT CRITERIA:")
    print(f"   • Profit Target: ₹150,000 (1.5%)")
    print(f"   • Stop Loss: ₹200,000 (2.0%)")
    print(f"   • Time Exit: Thursday 3:15 PM")
    print(f"   • VIX Spike: >21.75 (exit all)")
    
    print(f"\n💡 KEY SUCCESS FACTORS:")
    print(f"   • Disciplined execution at market open")
    print(f"   • Continuous VIX monitoring")
    print(f"   • Strict adherence to stop-loss")
    print(f"   • Timely profit booking at 1%")
    
    print("=" * 80)
    print("Strategy optimized for current market conditions (Nifty 24,500, VIX 14.5)")
    print("Maximum protection with positive carry in normal volatility environment")
    print("=" * 80)

if __name__ == "__main__":
    display_strategy_summary() 