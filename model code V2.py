
#This model was made with Nvidia as an example, so we use the market sentiment towards Nvidia
#we also used news from real life to explain our project questions and drive the thematics of whether we could predict with stochastic probability the trend in stocks
#Furthermore we used data and information on the stock from 2022, to analyze the different option and outcome with random occurence of events
#The plots found will serve as blueprint for whart couldve happened if events occured in a random manner or if some of these events did not happen
import random
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(None) 
random.seed()




#stock name used: Nvidia 2022, pfizer 2016, Exxon mobile 2007
#initialize the values, these should be found on yahoo finance and provided in the slides
# Define model parameters
T = 17 # Time horizon in years
N = 252 # Number of steps (trading days)
dt = T / N  # Time increment
Starting_Year=2007 #starting year
# for example google: price 2010
# Parameters for stock dynamics
r = 0.05 # Risk-free rate
S0 = 87 # Initial stock price
kappa = 2  # Mean reversion speed of volatility
theta = 0.04 # Long-term mean of variance
eta = 0.4# Volatility of variance
v0 = 0.1846 # Initial variance (volatility squared)
alpha = 0.1  # Speed of mean reversion for jump intensity
beta = 1.88# Long-term mean jump intensity
delta = 0.15 # Volatility of jump intensity

# Jump parameters
jump_mean = 0.03  # Mean jump size for stock price
jump_vol = 0.06 # Volatility of jump size
vol_jump_mean = 0.04  # Mean jump size for volatility
vol_jump_vol = 0.15 # Volatility of jump size
#we found it that it was best to put this dictionary here instead of attaching another file to the document
#i hope this doesnt make it hard to read and understand
events ={

    "AI Surge ": {"Sector": "Technology", "Magnitude": 1.0, "Sentiment": "Positive"},
    "Data Center Growth ": {"Sector": "Technology", "Magnitude": 1.0, "Sentiment": "Positive"},
    "Competition Concerns ": {"Sector": "Technology", "Magnitude": 0.41, "Sentiment": "Negative"},
    "Cryptocurrency Mining Bust ": {"Sector": "Technology", "Magnitude": 0.84, "Sentiment": "Negative"},
    "Stock Split ": {"Sector": "Technology", "Magnitude": 0.73, "Sentiment": "Positive"},
    "Acquisition of Array BioPharma ": {"Sector": "Health", "Magnitude": 0.73, "Sentiment": "Positive"},
    "Emergency Use Authorization for COVID-19 Vaccine ": {"Sector": "Health", "Magnitude": 1.0, "Sentiment": "Positive"},
    "Record COVID-19 Vaccine Sales ": {"Sector": "Health", "Magnitude": 1.0, "Sentiment": "Positive"},
    "Patent Infringement Lawsuits ": {"Sector": "Health", "Magnitude": 0.47, "Sentiment": "Negative"},
    "Launch of Paxlovid ": {"Sector": "Health", "Magnitude": 0.73, "Sentiment": "Positive"},
    "Revenue Miss in Q3 Earnings ": {"Sector": "Health", "Magnitude": 0.47, "Sentiment": "Negative"},
    "Acquisition of Seagen ": {"Sector": "Health", "Magnitude": 0.73, "Sentiment": "Positive"},
    "Declining COVID-19 Vaccine Demand": {"Sector": "Health", "Magnitude": 0.8, "Sentiment": "Negative"},
    "COVID-19 Lawsuit Win Against Moderna ": {"Sector": "Health", "Magnitude": 0.37, "Sentiment": "Positive"},
    "Weak Revenue Guidance for 2024 ": {"Sector": "Health", "Magnitude": 0.51, "Sentiment": "Negative"},
    "Record Profits Amid Rising Oil Prices ": {"Sector": "Energy", "Magnitude": 1.0, "Sentiment": "Positive"},
    "Gulf of Mexico Spill Scrutiny ": {"Sector": "Energy", "Magnitude": 0.8, "Sentiment": "Negative"},
    "Declining Oil Prices Impact": {"Sector": "Energy", "Magnitude": 0.8, "Sentiment": "Negative"},
    "Climate Change Shareholder Vote (2017)": {"Sector": "Energy", "Magnitude": 0.47, "Sentiment": "Negative"},
    "Guyana Oil Discoveries ": {"Sector": "Energy", "Magnitude": 1.0, "Sentiment": "Positive"},
    "COVID-19 Pandemic Losses ": {"Sector": "Energy", "Magnitude": 0.1, "Sentiment": "Negative"},
    "Shareholder Activist Victory ": {"Sector": "Energy", "Magnitude": 0.37, "Sentiment": "Positive"},
    "Ukraine War and Energy Crisis ": {"Sector": "Energy", "Magnitude": 0.73, "Sentiment": "Positive"},
    "Pioneer Natural Resources Acquisition ": {"Sector": "Energy", "Magnitude": 1.0, "Sentiment": "Positive"},
    "Carbon Capture and EV Investments ": {"Sector": "Energy", "Magnitude": 0.73, "Sentiment": "Positive"},
    "Positive Sentiment from AI Boom ": {"Sector": "Technology", "Magnitude": 0.91, "Sentiment": "Positive"},
    "Resilience Amid Market Challenges": {"Sector": "Technology", "Magnitude": 0.83, "Sentiment": "Positive"},
    "Product Releases and Market Optimism": {"Sector": "Technology", "Magnitude": 0.91, "Sentiment": "Positive"},
    "Strategic Partnerships": {"Sector": "Technology", "Magnitude": 0.85, "Sentiment": "Positive"},
    "Mixed Sentiment in Early": {"Sector": "Technology", "Magnitude": 0.61, "Sentiment": "Positive"}
}




# States: 0 = Bullish, 1 = Neutral, 2 = Bearish, since index start at 0, we took advantage to make use of that intuitively
market_sentiment_probs = np.array([[0.85, 0.1, 0.05],  # Bullish -> Bullish, Neutral, Bearish
                                   [0.15, 0.7, 0.15],    # Neutral -> Bullish, Neutral, Bearish
                                   [0.05, 0.2, 0.75]])   # Bearish -> Bullish, Neutral, Bearish
market_states = ["Bullish", "Neutral", "Bearish"]
initial_sentiment = 2  #initialize the sentiment by starting with a neutral market

#now we making the functions needed for the simulation we want to produce
# CIR process for stochastic jump intensity
def simulate_jump_intensity(lambda_t, alpha, beta, delta, dt):
    dZ = np.random.normal(0, np.sqrt(dt))
    return max(lambda_t + alpha * (beta - lambda_t) * dt + delta * np.sqrt(lambda_t) * dZ, 0)

# Heston process for variance with jumps
def simulate_variance(v_t, kappa, theta, eta, jump_intensity, vol_jump_mean, vol_jump_vol, dt):
    dW = np.random.normal(0, np.sqrt(dt))
    jump = np.random.normal(vol_jump_mean, vol_jump_vol) if np.random.rand() < jump_intensity * dt else 0
    v_t_new = max(v_t + kappa * (theta - v_t) * dt + eta * np.sqrt(v_t) * dW + jump, 0)
    return v_t_new

# Stock price dynamics with jump diffusion and market sentiment impact
def simulate_stock_price(S_t, v_t_new, lambda_t, jump_mean, jump_vol, market_state, dt,impact_str):
    dB = np.random.normal(0, np.sqrt(dt))
    jump = np.random.normal(jump_mean, jump_vol) if np.random.rand() < lambda_t * dt else 0
   
    # Adjust drift term based on market sentiment
    if market_state == 0:  # Bullish
        drift = r + 0.02 + impact_str  # Adding positive sentiment
    elif market_state == 2:  # Bearish
        drift = r - 0.02 - impact_str  # Adding negative sentiment
    else:
        drift = r

    S_t_new = S_t * (1 + drift * dt + np.sqrt(v_t_new) * dB + jump)
    return S_t_new
   
# Markov process for market sentiment transitions
def update_market_sentiment(current_state, market_sentiment_probs):
    return np.random.choice([0, 1, 2], p=market_sentiment_probs[current_state])

# News function to alter market sentiment probabilities
# Store the original probabilities as a baseline



def generate_news(target_sector, dictionary):

    news_event = random.choice(list(dictionary.keys()))
    news = dictionary[news_event]

    # Extract news attributes
    news_sector = news["Sector"]
    impact_strength = news["Magnitude"]
    sentiment = news["Sentiment"]

    # Start with a copy of the original probabilities
    new_probs = market_sentiment_probs.copy()

    if news_sector == target_sector:  # Relevant news
        if sentiment == "Positive" and impact_strength > 0.5:
            # Increase probabilities for Bullish state
            new_probs[0] = [0.9, 0.05,0.05 ]  # Fully bullish
            new_probs[1] = [0.8, 0.15, 0.05]
            new_probs[2] = [0.8, 0, 0.2]
        elif sentiment == "Negative" and impact_strength > 0.5:
            # Increase probabilities for Bearish state
            new_probs[0] = [0.9, 0.05, 0.05]
            new_probs[1] = [0.05, 0.15, 0.8]
            new_probs[2] = [0.05, 0.05, 0.9]
    else:  # Irrelevant news
        impact_strength = 0  # No impact
        new_probs = market_sentiment_probs.copy()

    return news_event, impact_strength, new_probs, news_sector, sentiment

#######################################################
#start of the simulation here
#=============================================================================

S_values = [S0]
V_values = [v0]
Lambda_values = [beta]  # Start jump intensity at long-term mean
Sentiment_values = [initial_sentiment]
News_events = []  # Keep track of news events and their effects
number = 5
colors=['blue', 'green', 'red','orange']
# Start with a copy of the original probabilities
current_market_sentiment_probs = market_sentiment_probs.copy()


all_stock_prices = []
all_sentiment_values = []  # Store sentiment values for each simulation
for i in range(1,N):
    if i % 20 == 0:  # Here 20 is the number of steps between each news but this number can be altered 
        news_type, impact_strength, updated_probs, news_sector, news_sentiment = generate_news('Energy', events)
        News_events.append((i * dt, news_type, impact_strength, news_sector, news_sentiment))
        current_market_sentiment_probs = updated_probs  # Update probabilities based on news
    else:
        impact_strength = 0  # No news impact on this iteration

    # Update market sentiment
    current_sentiment = update_market_sentiment(Sentiment_values[-1], current_market_sentiment_probs)
    Sentiment_values.append(current_sentiment)
    
for j in range(1, number):
    S_values = [S0]
    V_values = [v0]
    Lambda_values = [beta]  # Start jump intensity at long-term mean
    Sentiment_values = [initial_sentiment]
    News_events = []  # Keep track of news events and their effects



    for k in range(1, N):
  

        # Update market sentiment
        current_sentiment = update_market_sentiment(Sentiment_values[-1], current_market_sentiment_probs)
        Sentiment_values.append(current_sentiment)

        # Update jump intensity using CIR process
        lambda_t = simulate_jump_intensity(Lambda_values[-1], alpha, beta, delta, dt)
        Lambda_values.append(lambda_t)

        # Update variance using Heston process with jumps
        v_t = simulate_variance(V_values[-1], kappa, theta, eta, lambda_t, vol_jump_mean, vol_jump_vol, dt)
        V_values.append(v_t)

        # Update stock price with jump-diffusion and sentiment-based drift
        S_t = simulate_stock_price(S_values[-1], v_t, lambda_t, jump_mean, jump_vol, current_sentiment, dt, impact_strength)
        S_values.append(S_t)

    # Add stock prices and sentiment values for this simulation to the list
    all_stock_prices.append(np.array(S_values))
    all_sentiment_values.append(np.array(Sentiment_values))

# Plot all stock prices and market sentiment in one figure
time = np.linspace(Starting_Year, Starting_Year+T, N)

plt.figure(figsize=(15, 10))

# Plot all stock prices
plt.subplot(2, 1, 1)  # Create 2 subplots, 1 for stock prices and 1 for sentiment
for idx, S_values in enumerate(all_stock_prices):
    color_given = colors[idx % len(colors)]  # Cycle through colors
    plt.plot(time, S_values, label=f"Simulation {idx+1}", color=color_given)

plt.title("Stock Price Dynamics from Multiple Simulations")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()

# Plot market sentiment
plt.subplot(2, 1, 2)  # Plot in second subplot

plt.plot(time, Sentiment_values, label="Market Sentiment", color="purple")
plt.title("Market Sentiment Over Time")
plt.xlabel("Time")
plt.ylabel("Sentiment State (0=Bullish, 1=Neutral, 2=Bearish)")
plt.yticks([0, 1, 2], ["Bullish", "Neutral", "Bearish"])
plt.legend()

plt.tight_layout()
plt.show()


# Print news events
for event in News_events:     #only print relevant information but we can tweak this and give us all the news that occurs at this time
    if event[3] == 'Energy':   #remove this if statement to print out all the news
        print(f"Time: {event[0]:.2f}, News: {event[1]}, sector: {event[3]}, nature: {event[4]}, Impact Strength: {event[2]:.2f} ")
