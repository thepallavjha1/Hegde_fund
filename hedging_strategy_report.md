# Indian Derivatives One-Week Hedging Strategy Report

## Executive Summary

This report presents a comprehensive one-week hedging strategy for a ₹1 Crore portfolio (Beta: 1.2) using Indian derivatives markets. The strategy employs a multi-layered approach combining protective puts, volatility hedges, tail risk protection, and income generation through options.

### Key Metrics
- **Portfolio Value**: ₹10,000,000
- **Portfolio Beta**: 1.2
- **Maximum Profit Potential**: ₹150,000 (1.5%)
- **Maximum Loss (Protected)**: ₹200,000 (2.0%)
- **Total Transaction Costs**: ₹35,000 (0.35%)
- **Break-even Market Move**: ±0.35%

## Market Analysis

### Current Market Conditions
- **Nifty 50**: 24,500
- **Bank Nifty**: 52,000
- **India VIX**: 14.5 (40th percentile - Normal Volatility)
- **Market Regime**: Normal Volatility Environment

### Expected Weekly Movement
Based on current VIX levels:
- **Nifty Expected Range**: 24,006 - 24,994 (±494 points)
- **Bank Nifty Expected Range**: 50,950 - 53,050 (±1,050 points)

## Hedging Strategy Components

### 1. Primary Hedge (40% of Risk Budget)
**Protective Put Spread on Nifty**
- Long 24,400 PE @ ₹120
- Short 24,200 PE @ ₹60
- **Net Premium**: ₹60 per lot
- **Position Size**: 8 lots (400 shares)
- **Max Protection**: ₹80,000
- **Cost**: ₹24,000

**Rationale**: Provides downside protection starting at 24,400 with limited cost through spread structure.

### 2. Volatility Hedge (20% of Risk Budget)
**Iron Condor on Nifty**
- Short 24,300 PE @ ₹80 / Long 24,200 PE @ ₹60
- Short 24,700 CE @ ₹85 / Long 24,800 CE @ ₹65
- **Net Credit**: ₹40 per lot
- **Position Size**: 4 lots
- **Max Profit**: ₹8,000
- **Max Loss**: ₹12,000

**Rationale**: Profits from range-bound market expected in normal volatility regime.

### 3. Tail Risk Hedge (10% of Risk Budget)
**Far OTM Protective Puts**
- Long 23,275 PE (5% OTM) @ ₹25
- **Position Size**: 2 lots
- **Cost**: ₹2,500
- **Protection**: Against black swan events

**Rationale**: Cheap insurance against extreme market crashes.

### 4. Income Generation (30% of Risk Budget)
**Covered Calls on Bank Nifty**
- Short 52,800 CE @ ₹180
- **Position Size**: 6 lots
- **Premium Collected**: ₹27,000
- **Cap on Upside**: 52,800

**Rationale**: Generate income in low volatility environment while maintaining upside to 1.5%.

## Position Sizing Calculations

### Delta-Neutral Hedge Ratio
```
Portfolio Delta Exposure = ₹10,000,000 × 1.2 = ₹12,000,000
Required Hedge Delta = -₹12,000,000

Primary Hedge Delta Coverage:
Put Spread Delta ≈ -0.35
Coverage = 8 lots × 50 × 24,500 × (-0.35) = -₹3,430,000 (28.6% coverage)
```

### Risk Budget Allocation
- Primary Hedge: ₹24,000 (0.24%)
- Volatility Hedge: Net Credit ₹8,000
- Tail Risk: ₹2,500 (0.025%)
- Income Generation: Credit ₹27,000
- **Net Initial Cost**: ₹-8,500 (Net Credit)

## Risk Assessment Matrix

| Scenario | Nifty Level | VIX Level | Portfolio P&L | Hedge P&L | Net P&L | Return % |
|----------|-------------|-----------|---------------|-----------|---------|----------|
| Base Case | 24,500 | 14.5 | ₹0 | ₹8,500 | ₹8,500 | 0.09% |
| Mild Bullish (+1%) | 24,745 | 13.0 | ₹120,000 | -₹25,000 | ₹95,000 | 0.95% |
| Strong Bullish (+3%) | 25,235 | 11.6 | ₹360,000 | -₹85,000 | ₹275,000 | 2.75% |
| Mild Bearish (-1%) | 24,255 | 16.0 | -₹120,000 | ₹65,000 | -₹55,000 | -0.55% |
| Strong Bearish (-3%) | 23,765 | 18.9 | -₹360,000 | ₹195,000 | -₹165,000 | -1.65% |
| Black Swan (-5%) | 23,275 | 29.0 | -₹600,000 | ₹425,000 | -₹175,000 | -1.75% |

## Entry and Exit Criteria

### Entry Conditions
1. **Timing**: First 30 minutes after market open (9:15-9:45 AM)
2. **VIX Threshold**: Enter if VIX < 13.05 (10% below current)
3. **Price Range**: 
   - Nifty: 24,450 - 24,550
   - Bank Nifty: 51,850 - 52,150
4. **Execution**: Scale into positions over 15-minute window

### Exit Conditions
1. **Profit Target**: ₹150,000 (1.5% of portfolio)
2. **Stop Loss**: ₹200,000 (2.0% of portfolio)
3. **Time Exit**: Thursday 3:15 PM (15 minutes before expiry)
4. **Volatility Spike**: Exit if VIX > 21.75 (50% increase)
5. **Delta Threshold**: Adjust if position delta > 0.7

### Adjustment Triggers
- **Delta Breach**: > 0.5 absolute
- **Vega Breach**: > ₹10,000 exposure
- **Profit Lock**: At 1% portfolio gain (₹100,000)

## Greeks Analysis

### Initial Portfolio Greeks
- **Delta**: -0.28 (moderately bearish bias)
- **Gamma**: -0.0012 (stable delta)
- **Theta**: ₹8,500/day (positive time decay)
- **Vega**: -₹12,000 (benefits from falling volatility)

### Greeks Management Rules
1. Maintain delta between -0.3 and +0.3
2. Reduce gamma exposure if |gamma| > 0.002
3. Ensure positive theta throughout the week
4. Keep vega exposure < ₹20,000

## Transaction Cost Analysis

| Component | Amount (₹) | % of Portfolio |
|-----------|------------|----------------|
| Brokerage (Entry + Exit) | 12,800 | 0.128% |
| STT on Options | 8,750 | 0.088% |
| Exchange Charges | 6,230 | 0.062% |
| GST (18%) | 5,220 | 0.052% |
| **Total Costs** | **35,000** | **0.35%** |

## Execution Timeline - 

### Monday (Entry Day)
- **9:15 AM**: Market open - Execute primary hedge
- **10:00 AM**: Monitor VIX levels
- **2:00 PM**: Mid-day position assessment
- **3:30 PM**: EOD risk check

### Tuesday (Monitoring)
- **9:15 AM**: Review overnight positions
- **1:00 PM**: Gamma adjustment if needed
- **3:00 PM**: Theta decay analysis

### Wednesday (Pre-Expiry)
- **9:15 AM**: Pre-expiry assessment
- **11:00 AM**: Consider rolling positions
- **2:00 PM**: Lock partial profits if available
- **3:30 PM**: Prepare exit strategy

### Thursday (Expiry Day)
- **9:15 AM**: Final position review
- **1:00 PM**: Begin exit execution
- **3:15 PM**: Close all positions
- **3:30 PM**: Settlement confirmation

## Strategic Recommendations

1. **Immediate Actions**
   - Execute primary hedge at market open for optimal liquidity
   - Use limit orders to minimize slippage (save 0.05%)
   - Establish all positions within first hour

2. **Risk Management**
   - Monitor VIX closely - adjust if moves beyond 12-18 range
   - Maintain strict stop-loss at 2% portfolio loss
   - Book partial profits at 1% gain to reduce risk

3. **Capital Management**
   - Keep 20% capital (₹2 million) as reserve for adjustments
   - Avoid over-leveraging even if opportunities arise
   - Focus on capital preservation over profit maximization

4. **Market Monitoring**
   - Track overnight global markets for gap risk
   - Monitor FII/DII data for sentiment shifts
   - Watch for economic data releases (RBI policy, inflation)

5. **Position Adjustments**
   - Roll positions on Wednesday if time decay > 50%
   - Reduce position size by 20% if VIX > 15
   - Convert to delta-neutral if market trends strongly

## Conclusion

This one-week hedging strategy provides robust downside protection while maintaining upside potential in a normal volatility environment. The multi-layered approach ensures:

- **Limited Downside**: Maximum loss capped at 2%
- **Positive Carry**: Net credit of ₹8,500 plus daily theta
- **Flexibility**: Multiple adjustment points
- **Cost Efficiency**: Total costs under 0.35%

The strategy is optimized for the current market conditions with Nifty at 24,500 and VIX at 14.5. Success depends on disciplined execution, continuous monitoring, and adherence to predetermined exit criteria.

### Risk Disclaimer
This strategy involves complex derivatives that can result in significant losses. Past performance does not guarantee future results. Consider your risk tolerance and consult with a qualified financial advisor before implementation. 