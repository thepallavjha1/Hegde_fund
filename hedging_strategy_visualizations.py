"""
Visualization Tools for Indian Derivatives Hedging Strategy
Creates professional charts for strategy analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set professional plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class HedgingStrategyVisualizer:
    """Create professional visualizations for hedging strategy"""
    
    def __init__(self, spot_price: float, strikes: dict, portfolio_value: float):
        self.spot_price = spot_price
        self.strikes = strikes
        self.portfolio_value = portfolio_value
        
    def plot_payoff_diagram(self, strategy_name: str, legs: list):
        """Create payoff diagram for option strategies"""
        # Price range for x-axis
        price_range = np.linspace(self.spot_price * 0.85, self.spot_price * 1.15, 200)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Calculate individual leg payoffs
        total_payoff = np.zeros_like(price_range)
        
        for leg in legs:
            strike = leg['strike']
            premium = leg['premium']
            position = leg['position']  # 1 for long, -1 for short
            option_type = leg['type']  # 'CE' or 'PE'
            quantity = leg['quantity']
            
            if option_type == 'CE':
                intrinsic = np.maximum(price_range - strike, 0)
            else:  # PE
                intrinsic = np.maximum(strike - price_range, 0)
            
            leg_payoff = position * (intrinsic - premium) * quantity
            total_payoff += leg_payoff
            
            # Plot individual legs
            ax1.plot(price_range, leg_payoff, '--', alpha=0.5, linewidth=1,
                    label=f"{leg['name']}")
        
        # Plot total payoff
        ax1.plot(price_range, total_payoff, 'b-', linewidth=3, label='Net Payoff')
        
        # Highlight profit/loss regions
        ax1.fill_between(price_range, 0, total_payoff, 
                        where=total_payoff > 0, alpha=0.3, color='green', 
                        label='Profit Zone')
        ax1.fill_between(price_range, 0, total_payoff, 
                        where=total_payoff < 0, alpha=0.3, color='red', 
                        label='Loss Zone')
        
        # Mark current spot price
        ax1.axvline(x=self.spot_price, color='black', linestyle=':', linewidth=2,
                   label=f'Spot: {self.spot_price:.0f}')
        
        # Mark breakeven points
        breakeven_indices = np.where(np.diff(np.sign(total_payoff)))[0]
        for idx in breakeven_indices:
            breakeven_price = price_range[idx]
            ax1.axvline(x=breakeven_price, color='orange', linestyle='--', alpha=0.7)
            ax1.text(breakeven_price, ax1.get_ylim()[1]*0.9, 
                    f'BE: {breakeven_price:.0f}', rotation=90, ha='right')
        
        # Formatting
        ax1.set_title(f'{strategy_name} - Payoff Diagram', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Nifty Price at Expiry', fontsize=12)
        ax1.set_ylabel('Profit/Loss (₹)', fontsize=12)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linewidth=1)
        
        # Add key metrics
        max_profit = np.max(total_payoff)
        max_loss = np.min(total_payoff)
        ax1.text(0.02, 0.95, f'Max Profit: ₹{max_profit:,.0f}', 
                transform=ax1.transAxes, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax1.text(0.02, 0.88, f'Max Loss: ₹{max_loss:,.0f}', 
                transform=ax1.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        # Probability distribution (simplified normal distribution)
        prob_dist = np.exp(-0.5 * ((price_range - self.spot_price) / (self.spot_price * 0.02))**2)
        prob_dist = prob_dist / np.max(prob_dist) * 0.3
        ax2.fill_between(price_range, 0, prob_dist, alpha=0.5, color='gray')
        ax2.set_xlabel('Nifty Price', fontsize=12)
        ax2.set_ylabel('Probability', fontsize=10)
        ax2.set_ylim(0, 0.4)
        ax2.axvline(x=self.spot_price, color='black', linestyle=':', linewidth=2)
        
        plt.tight_layout()
        return fig
    
    def plot_greeks_heatmap(self, strikes: list, spot_range: tuple, vol_range: tuple):
        """Create heatmap showing Greeks sensitivity"""
        spot_prices = np.linspace(spot_range[0], spot_range[1], 20)
        volatilities = np.linspace(vol_range[0], vol_range[1], 20)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Greeks to plot
        greeks = ['Delta', 'Gamma', 'Vega', 'Theta']
        
        for idx, (ax, greek) in enumerate(zip(axes.flatten(), greeks)):
            # Create sample heatmap data (simplified)
            heatmap_data = np.random.randn(len(volatilities), len(spot_prices))
            
            # Adjust data based on Greek characteristics
            if greek == 'Delta':
                for i, spot in enumerate(spot_prices):
                    heatmap_data[:, i] = (spot - self.spot_price) / self.spot_price
            elif greek == 'Gamma':
                for i, spot in enumerate(spot_prices):
                    heatmap_data[:, i] = np.exp(-((spot - self.spot_price)**2) / (2 * (self.spot_price * 0.02)**2))
            elif greek == 'Vega':
                for j, vol in enumerate(volatilities):
                    heatmap_data[j, :] = vol / 15
            elif greek == 'Theta':
                heatmap_data = -np.abs(heatmap_data) * 0.5
            
            # Create heatmap
            im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', 
                          extent=[spot_range[0], spot_range[1], vol_range[0], vol_range[1]])
            ax.set_title(f'{greek} Sensitivity', fontsize=14, fontweight='bold')
            ax.set_xlabel('Spot Price', fontsize=12)
            ax.set_ylabel('Implied Volatility (%)', fontsize=12)
            
            # Add current position marker
            ax.plot(self.spot_price, 14.5, 'ko', markersize=10)
            ax.text(self.spot_price, 14.5, 'Current', ha='center', va='bottom')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f'{greek} Value', fontsize=10)
        
        plt.suptitle('Greeks Sensitivity Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_risk_scenarios(self, scenarios_df: pd.DataFrame):
        """Create risk scenario analysis chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Waterfall chart for P&L by scenario
        scenarios = scenarios_df['Scenario'].values
        net_pnl = scenarios_df['Net_PnL'].values
        colors = ['green' if pnl > 0 else 'red' for pnl in net_pnl]
        
        bars = ax1.bar(range(len(scenarios)), net_pnl, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(scenarios)))
        ax1.set_xticklabels(scenarios, rotation=45, ha='right')
        ax1.set_title('Net P&L by Market Scenario', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Profit/Loss (₹)', fontsize=12)
        ax1.axhline(y=0, color='black', linewidth=1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, net_pnl):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'₹{value:,.0f}', ha='center', 
                    va='bottom' if height > 0 else 'top')
        
        # Portfolio vs Hedge P&L comparison
        portfolio_pnl = scenarios_df['Portfolio_PnL'].values
        hedge_pnl = scenarios_df['Hedge_PnL'].values
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, portfolio_pnl, width, label='Portfolio P&L', alpha=0.7)
        bars2 = ax2.bar(x + width/2, hedge_pnl, width, label='Hedge P&L', alpha=0.7)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenarios, rotation=45, ha='right')
        ax2.set_title('Portfolio vs Hedge Performance', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Profit/Loss (₹)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linewidth=1)
        
        plt.tight_layout()
        return fig
    
    def plot_execution_timeline(self):
        """Create execution timeline visualization"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Timeline data
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
        times = ['9:15 AM', '10:00 AM', '2:00 PM', '3:30 PM']
        
        # Create timeline
        for i, day in enumerate(days):
            y_pos = i
            ax.barh(y_pos, 1, left=0, height=0.5, alpha=0.3, color='lightblue')
            ax.text(-0.1, y_pos, day, ha='right', va='center', fontsize=12, fontweight='bold')
            
            # Add events
            if day == 'Monday':
                events = [
                    (0.1, 'Market Open: Execute Primary Hedge'),
                    (0.3, 'Monitor VIX levels'),
                    (0.7, 'Mid-day position check'),
                    (0.9, 'EOD risk assessment')
                ]
            elif day == 'Tuesday':
                events = [
                    (0.1, 'Review overnight positions'),
                    (0.5, 'Gamma adjustment if needed'),
                    (0.9, 'Theta decay analysis')
                ]
            elif day == 'Wednesday':
                events = [
                    (0.1, 'Pre-expiry assessment'),
                    (0.3, 'Consider rolling positions'),
                    (0.6, 'Lock partial profits'),
                    (0.9, 'Prepare exit strategy')
                ]
            else:  # Thursday
                events = [
                    (0.1, 'Final position review'),
                    (0.5, 'Execute exit strategy'),
                    (0.85, 'Close all positions by 3:15 PM'),
                    (0.95, 'Settlement')
                ]
            
            for x_pos, event in events:
                ax.plot(x_pos, y_pos, 'ro', markersize=8)
                ax.text(x_pos, y_pos + 0.3, event, rotation=45, 
                       ha='left', va='bottom', fontsize=9)
        
        ax.set_xlim(-0.2, 1.1)
        ax.set_ylim(-0.5, len(days) - 0.5)
        ax.set_title('One-Week Hedging Strategy Execution Timeline', fontsize=16, fontweight='bold')
        ax.set_xlabel('Day Progress', fontsize=12)
        ax.set_yticks(range(len(days)))
        ax.set_yticklabels([])
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add legend
        red_patch = mpatches.Patch(color='red', label='Key Action Points')
        blue_patch = mpatches.Patch(color='lightblue', alpha=0.3, label='Trading Day')
        ax.legend(handles=[red_patch, blue_patch], loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_report(self, strategy_data: dict):
        """Create multi-page comprehensive visual report"""
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 28))
        
        # Page 1: Strategy Overview
        gs = fig.add_gridspec(6, 3, hspace=0.4, wspace=0.3)
        
        # Title
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.text(0.5, 0.5, 'Indian Derivatives One-Week Hedging Strategy Report', 
                     ha='center', va='center', fontsize=24, fontweight='bold',
                     transform=ax_title.transAxes)
        ax_title.axis('off')
        
        # Key metrics
        ax_metrics = fig.add_subplot(gs[1, :])
        metrics_text = f"""
Portfolio Value: ₹{strategy_data['portfolio_value']:,.0f} | Beta: {strategy_data['portfolio_beta']} | 
Max Profit: ₹{strategy_data['max_profit']:,.0f} | Max Loss: ₹{strategy_data['max_loss']:,.0f} | 
Transaction Costs: ₹{strategy_data['transaction_costs']:,.0f} ({strategy_data['cost_percentage']:.2f}%)
        """
        ax_metrics.text(0.5, 0.5, metrics_text, ha='center', va='center', 
                       fontsize=14, transform=ax_metrics.transAxes,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
        ax_metrics.axis('off')
        
        # Save figure
        plt.savefig('hedging_strategy_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return 'Report saved as hedging_strategy_report.png'

# Example usage
def create_sample_visualizations():
    """Create sample visualizations for the hedging strategy"""
    
    # Initialize visualizer
    viz = HedgingStrategyVisualizer(
        spot_price=24500,
        strikes={'atm': 24500, 'otm_put': 24300, 'otm_call': 24700},
        portfolio_value=10000000
    )
    
    # 1. Put Spread Payoff Diagram
    put_spread_legs = [
        {'name': 'Long 24400 PE', 'strike': 24400, 'premium': 120, 
         'position': 1, 'type': 'PE', 'quantity': 50},
        {'name': 'Short 24200 PE', 'strike': 24200, 'premium': 60, 
         'position': -1, 'type': 'PE', 'quantity': 50}
    ]
    
    fig1 = viz.plot_payoff_diagram('Protective Put Spread', put_spread_legs)
    fig1.savefig('put_spread_payoff.png', dpi=300, bbox_inches='tight')
    
    # 2. Iron Condor Payoff
    iron_condor_legs = [
        {'name': 'Short 24300 PE', 'strike': 24300, 'premium': 80, 
         'position': -1, 'type': 'PE', 'quantity': 50},
        {'name': 'Long 24200 PE', 'strike': 24200, 'premium': 60, 
         'position': 1, 'type': 'PE', 'quantity': 50},
        {'name': 'Short 24700 CE', 'strike': 24700, 'premium': 85, 
         'position': -1, 'type': 'CE', 'quantity': 50},
        {'name': 'Long 24800 CE', 'strike': 24800, 'premium': 65, 
         'position': 1, 'type': 'CE', 'quantity': 50}
    ]
    
    fig2 = viz.plot_payoff_diagram('Iron Condor', iron_condor_legs)
    fig2.savefig('iron_condor_payoff.png', dpi=300, bbox_inches='tight')
    
    # 3. Greeks Heatmap
    fig3 = viz.plot_greeks_heatmap(
        strikes=[24200, 24300, 24400, 24500, 24600, 24700, 24800],
        spot_range=(23500, 25500),
        vol_range=(10, 20)
    )
    fig3.savefig('greeks_heatmap.png', dpi=300, bbox_inches='tight')
    
    # 4. Risk Scenarios
    scenarios_data = {
        'Scenario': ['Base Case', 'Mild Bull', 'Strong Bull', 'Mild Bear', 'Strong Bear', 'Black Swan'],
        'Portfolio_PnL': [0, 100000, 300000, -100000, -300000, -500000],
        'Hedge_PnL': [0, -50000, -120000, 80000, 250000, 450000],
        'Net_PnL': [0, 50000, 180000, -20000, -50000, -50000]
    }
    scenarios_df = pd.DataFrame(scenarios_data)
    
    fig4 = viz.plot_risk_scenarios(scenarios_df)
    fig4.savefig('risk_scenarios.png', dpi=300, bbox_inches='tight')
    
    # 5. Execution Timeline
    fig5 = viz.plot_execution_timeline()
    fig5.savefig('execution_timeline.png', dpi=300, bbox_inches='tight')
    
    print("All visualizations created successfully!")
    return "Visualizations saved: put_spread_payoff.png, iron_condor_payoff.png, greeks_heatmap.png, risk_scenarios.png, execution_timeline.png"

if __name__ == "__main__":
    create_sample_visualizations() 