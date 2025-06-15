# %% [markdown]
# # Assignment 3
# 
# 
# ## Instructions
# 
# Please complete this Jupyter notebook and then convert it to a `.py` file called `assignment3.py`. Upload this file to Gradescope, and await feedback. 
# 
# You may submit as many times as you want up until the deadline. Only your latest submission counts toward your grade.
# 
# Some tests are hidden and some are visible. The outcome of the visible checks will be displayed to you immediately after you submit to Gradescope. The hidden test outcomes will be revealed after final scores are published. 
# 
# This means that an important part of any strategy is to **start early** and **lock in all the visible test points**. After that, brainstorm what the hidden checks could be and collaborate with your teammates.
# 

# %% [markdown]
# ### Problem 1
# 
# 
# Recall the derivation of the posterior
# \begin{align}
# \pi(\theta \mid y) 
# &\propto L(y \mid \theta) \pi(\theta) \\
# &\propto \underbrace{ \left\{ \theta^{-n/2}\exp\left[-\frac{\sum_i y_i^2}{2\theta} \right] \right\}}_{ \propto L(y \mid \theta)} \underbrace{\theta^{-(a+1)}\exp\left[ - b/\theta \right] }_{ \propto \pi(\theta)}  \\
# &= \theta^{-(a + n/2 + 1)} \exp\left[ - \frac{b  + ns/2}{\theta}\right]
# \end{align}
# 
# where $\theta > 0$ and $\pi(\theta) = \text{Inverse-Gamma}(a,b)$ and
# 
# $$
# L(y \mid \theta) \propto \theta^{-n/2}\exp\left[-\frac{\sum_i y_i^2}{2\theta} \right]
# $$
# 
# 1.
# 
# What is the natural logarithm of the normalizing constant of the final line? In other words, what do we have to divide $\theta^{-(a + n/2 + 1)} \exp\left[ - \frac{b  + ns/2}{\theta}\right]$ by so that it integrates to $1$? Then take the natural log of that. 
# 
# Stated differently, what is $\log \int_0^\infty \theta^{-(a + n/2 + 1)} \exp\left[ - \frac{b  + ns/2}{\theta}\right] \text{d} \theta$? 
# 
# Assume
#  - $a = 10$
#  - $b = 11$
#  - $n = 42$
#  - $s = 15$
#  
# 
# 
# Assign your answer to `log_norm_const`
# 
# NB1: if we didn't use the logarithm, the normalizing constant would be *way* too close to $0$.
# 
# 
# NB2: You're not doing calculus here. Rely on the fact that every normalized density integrates to $1$.

# %%
from scipy.special import gammaln
import numpy as np

alpha = 32
beta = 326

log_norm_const = gammaln(alpha - 1) - (alpha - 1) * np.log(beta)

# %% [markdown]
# 2. 
# 
# Are either of these dependent on the value of $\theta$? If yes, assign `True` to `dependent_on_theta`. Otherwise assign `False`

# %%
dependent_on_theta = False

# %% [markdown]
# ### Problem 2
# 
# 
# Assume the same model as the previous question except assume the mean of $y \mid \theta$ is now $\mu \neq 0$. You can continue to assume that $\mu$ is still known, it's just nonzero.
# 
# How do the derivations change? Adapt the derivations and upload a scanned copy of your work to Gradescope portal.
# 
# 

# %% [markdown]
# ### Problem 3
# 
# Sometimes picking the hyperparameters of a prior can be tricky if they don't have an easy interpretation. Here is a way to pick a prior that involves simulating data. If the data simulations look like you would expect, then the prior is a reasonable choice.
# 
# Assume the same model as the question one and assume we are dealing with medium-frequency (e.g. every five seconds) stock index percentage returns scaled by $100$. Choose an inverse gamma prior by simulating from the **prior predictive distribution.** The prior predictive distribution is
# 
# $$
# p(y) = \int L(y \mid \theta) \pi(\theta) \text{d}\theta.
# $$
# 
# 
# NB1: **do not look at any data before doing this!** You will all have different priors!
# 
# NB2: it might take you a few iterations of all these subquestions to find hyperparameters that you like. 

# %% [markdown]
# 1.
# 
# First, assign your chosen $a$ and $b$ hyperparameters to `prior_a` and `prior_b`. Please restrict your attention to $a > 2$ (I'll explain why in class).

# %%
prior_a = 2.1
prior_b = 500

# %% [markdown]
# 2.
# 
# Simulate $\theta^1, \theta^2, \ldots, \theta^{10,000}$ from the prior. Call these samples `prior_param_samples`
# 
# NB: we are using a *superscript* to denote iteration number.

# %%
from scipy.stats import invgamma

prior_param_samples = invgamma.rvs(a=prior_a, scale=prior_b, size=10_000)

# %% [markdown]
# 2.
# 
# For each parameter sample, simulate $100$ stock returns from the likelihood. Arrange your simulations as one super long numpy array. Call it `prior_predic_samps`.
# 
# NB: For parameter sample $i$, you have 
# 
# $$
# y_1, \ldots, y_{100} \mid \theta^i \sim \text{Normal}(0, \theta^i)
# $$
# 
# NB2: Each $\theta^i$ is the **variance** not the standard deviation.

# %%
prior_predic_samps = np.random.normal(
    loc=0,
    scale=np.sqrt(prior_param_samples)[:, np.newaxis],  # shape becomes (10000, 1)
    size=(10_000, 100)
).flatten()

# %% [markdown]
# 3.
# 
# Make a histogram of all your data samples. Upload a `.pdf` or a `.png` to Gradescope. Remember, this picture has to agree with your intuition about what stock returns could look like. Otherwise, your prior hyperparameters aren't a good choice!

# %% [markdown]
# import matplotlib.pyplot as plt
# 
# plt.figure(figsize=(10, 6))
# plt.hist(prior_predic_samps, bins=200, density=True, color='steelblue', edgecolor='black', range=(-125, 125))
# plt.title("Histogram of Prior Predictive Samples (Stock Returns)")
# plt.xlabel("Simulated Return")
# plt.ylabel("Density")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("prior_predictive_hist.png")
# plt.show()
# plt.close()

# %% [markdown]
# 4.
# 
# What is the difference between a prior predictive distribution and a posterior predictive distribution? What do they have in common? Upload your free response to Gradescope.

# %% [markdown]
# ### Problem 4
# 
# 
# Recall the derivation of the posterior
# $$
# \theta \mid y_1, \ldots, y_n \sim \text{Normal}\left( \bar{x}\left(\frac{\frac{n}{1}}{\frac{1}{b} + \frac{n}{1}} \right) + a\left(\frac{\frac{1}{b}}{\frac{1}{b} + \frac{n}{1}} \right) ,\frac{1}{\frac{1}{b} + \frac{n}{1} } \right)
# $$
# 
# where $\theta$ is the mean parameter, $\pi(\theta) = \text{Normal}(a,b)$ and 
# 
# 
# The work was 
# $$
# L(y \mid \theta) \propto \exp\left[-\frac{\sum_i (y_i-\theta)^2}{2} \right]
# $$
# 
# \begin{align}
# \pi(\theta \mid y) 
# &\propto L(y \mid \theta) \pi(\theta) \\
# &\propto \exp\left[ -\frac{1}{2} \frac{\left(\theta - \text{post. mean} \right)^2}{ \text{post. var.}} \right]
# \end{align}
# 
# 
# 1.
# 
# What is the natural logarithm of the normalizing constant of the final line? In other words, what do we have to divide $\exp\left[ -\frac{1}{2} \frac{\left(\theta - \text{post. mean} \right)^2}{ \text{post. var.}} \right]$ by so that it integrates to $1$? Then take the natural log of that. 
# 
# Stated differently, what is $\log \int_{-\infty}^\infty \exp\left[ -\frac{1}{2} \frac{\left(\theta - \text{post. mean} \right)^2}{ \text{post. var.}} \right]\text{d} \theta$? 
# 
# 
# Assign your answer to `log_norm_const2`
# 
# NB: You're not doing calculus here. Rely on the fact that every normalized density integrates to $1$.
# 
# 
# Assume
#  - $a = 10$
#  - $b = 11$
#  - $n = 42$
#  - $\bar{x} = 15$
# 
# 

# %%
a = 10
b = 11
n = 42
x_bar = 15

post_var = 1 / (1/b + n)

log_norm_const2 = 0.5 * np.log(2 * np.pi * post_var)

# %% [markdown]
# ### Problem 5
# 
# 
# Assume the same model as the previous question except assume the variance of $y \mid \theta$ is now $\sigma^2 \neq 1$. How do the derivations change? Adapt the derivations and upload a scanned copy of your work to Gradescope portal.
# 
# 

# %% [markdown]
# ### Problem 6
# 
# We will return to the model described in question one. Specifically, we will assume our data are normally distributed with mean $0$, and that we're only uncertain about the variance parameter. We will also use the prior hyperparameters we chose in an earlier problem!
# 
# Our data set will be intraday stock returns. 
# 
# 1.
# 
# Download and read in the data set `SPY-STK.csv`. Ignore every column except `bid_price_close` and `time`. These are prices of the S\&P 500 exchange traded fund recorded on March 26, 2024. Call your data set `stock_data` and store it as a `pandas` `DataFrame`. 
# 

# %%
import pandas as pd

stock_data = pd.read_csv("SPY-STK.csv", usecols=['bid_price_close', 'time'])

# %% [markdown]
# 2.
# 
# Calculate percent returns and make sure to scale them by $100$. Store them in a `pandas` `Series` called `one_day_returns`.

# %%
one_day_returns = stock_data['bid_price_close'].pct_change().dropna() * 100

# %% [markdown]
# 3.
# 
# Assign your Inverse Gamma posterior hyperparameters to `posterior_a` and `posterior_b`. Thenc reate an `scipy.stats.invgamma` for your posterior. Give it the right hyperparameters and call it `posterior` 

# %%
n = len(one_day_returns)
squared_sum = (one_day_returns ** 2).sum()

posterior_a = prior_a + n / 2
posterior_b = prior_b + squared_sum / 2

posterior = invgamma(a=posterior_a, scale=posterior_b)

# %% [markdown]
# 4. 
# 
# 
# Sample 10,000 single returns from the posterior predictive distribution. Make it a `numpy` array and call it `post_pred_samps`.

# %%
theta_samples = posterior.rvs(size=10_000)

post_pred_samps = np.random.normal(loc=0, scale=np.sqrt(theta_samples))

# %% [markdown]
# 5. 
# 
# 
# Use the posterior predictive samples and create two plots to show whether this model represents reality well. Use a histogram and a time series plot. 
# 
# Do the histograms look similar? Do the time-ordered observations look similar. What are the strengths and weaknesses of this model?
# 
# What you are doing now is called a **posterior predictive check**. 

# %% [markdown]
# # Plot 1: Histogram of posterior predictive returns
# plt.figure(figsize=(10, 4))
# plt.hist(post_pred_samps, bins=100, density=True, edgecolor='black')
# plt.title('Histogram of Posterior Predictive Returns')
# plt.xlabel('Return (%)')
# plt.ylabel('Density')
# plt.grid(True)
# plt.show()
# 
# # Plot 2: Time series of posterior predictive returns
# plt.figure(figsize=(10, 4))
# plt.plot(post_pred_samps[:500], linewidth=1)
# plt.title('Posterior Predictive Returns (First 500 Simulated Points)')
# plt.xlabel('Time Index')
# plt.ylabel('Return (%)')
# plt.grid(True)
# plt.show()


