import numpy
import logging
from scipy.stats import norm
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename='default.log', encoding='utf-8', level=logging.DEBUG)

# def calc_volatility(stock_price_hist: list):
#     # A simple way to calculate volatility is to use  std deviation
#     # https://www.wallstreetmojo.com/volatility-formula/
#     # Note: there seems to be a slight error in their calc, because it's slightly
#     # different than the one calculated by numpy.std() - e.g.
#     # their mean is 162.23, although it's actually 162.5036
#     # This measure seems to give higher volatility than expected
#     logging.debug('Initiating calculation of volatility - method 1')
#     mean = numpy.mean(stock_price_hist)
#     daily_volat = numpy.std(stock_price_hist)
#     annualized_volat = numpy.sqrt(252)*daily_volat  # 252 is no. of trading days in a year
#     an_volat_percent = 100 * annualized_volat / mean
#     # Need to divide by the mean because annualized_volat is in absolute terms
#     logging.info(f'Calculated annualized volatility (1) = {annualized_volat}')
#     return an_volat_percent


class Pricer:
    def __init__(self, time_to_mat, underlying_price, strike_price, stock_price_hist: list):
        self.time_to_mat = time_to_mat
        self.underlying_price = underlying_price
        self.strike_price = strike_price
        self.calculate = None   # this will be the calculate function from the child class.
        self.sigma = self.calc_volatility_log(stock_price_hist)
        # Using a static method to calculate the annualized volatility based on the stock price history.
        # The advantage of this is that the volatility will be calculated automatically when an instance is created
        # rather than having to call the method later on.

    @staticmethod
    def calc_volatility_log(stock_price_hist):
        # Using logarithmic returns
        # Log returns are a way of measuring the percentage change over time
        # https://www.macroption.com/historical-volatility-calculation/
        # Compared to online volatility data, this method seems more accurate
        logging.debug('Initiating calculation of volatility')
        log_returns = [numpy.log(stock_price_hist[i] / stock_price_hist[i - 1])
                       for i in range(1, len(stock_price_hist))]

        daily_volat = numpy.std(log_returns)
        annualized_volat = numpy.sqrt(252)*daily_volat  # 252 is no. of trading days in a year
        an_volat_percent = 100 * annualized_volat
        # No need to divide by the mean because annualized_volat is already
        # calculated as a ratio from the log_returns formula
        logging.info(f'Calculated annualized volatility (1) = {annualized_volat}')
        return annualized_volat

    def plot(self, step="default", k_min="default", k_max="default", t_min=1/12, t_max=2):
        if k_min == "default":
            k_min = round((self.underlying_price - 0)/2) # Default min value is half the underlying price
        elif isinstance(k_min, int):
            raise ValueError("Min strike price must be more an integer")
        elif k_min < 0:
            raise ValueError("Strike price cannot be negative!")
        if k_max == "default":
            k_max = round(self.underlying_price * 2) # Default max value is double the underlying price
        elif isinstance(k_max, int):
            raise ValueError("Max strike price must be more an integer")
        elif k_max < k_min:
            raise ValueError("Max value cannot be less than the min value!")
        if not (isinstance(t_min, float) or isinstance(t_min, int)):
            raise ValueError("Min time must be more a number")
        if not (isinstance(t_max, float) or isinstance(t_max, int)):
            raise ValueError("Max time must be more a number")
        if t_min <= 0:
            raise ValueError("Time to maturity cannot be zero or negative!")
        if t_max < t_min:
            raise ValueError("Max time to maturity cannot be less than the min value!")
        if step == "default":
            step = int((k_max - k_min)/10)
        elif step and step <= 0:
            raise ValueError("Number of steps must be more than 0")
        elif step and not isinstance(step, int):
            raise ValueError("Number of steps must be more an integer")

        def option_price(k, t):
            self.strike_price = k
            self.time_to_mat = t
            return self.calculate()  # returning a tuple: call, put

        fig1 = plt.figure()
        fig2 = plt.figure()
        ax = fig1.add_subplot(projection='3d')
        ax2 = fig2.add_subplot(projection='3d')
        k_range = numpy.arange(140, 200, 5)
        t_range = numpy.arange(0.5, 1.5, 0.1)
        X, Y = numpy.meshgrid(k_range, t_range)
        Z_call, Z_put = option_price(X, Y)
        # Z_put = option_price(X, Y)[1]

        surf = ax.plot_surface(X, Y, Z_call, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
        surf2 = ax2.plot_surface(X, Y, Z_put, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)

        ax.set_zlim(0, max(Z_call[0]))
        ax2.set_zlim(0, max(Z_put[0]))

        # Plot the surface.
        plt.show()


class BlackScholes(Pricer):
    def __init__(self, time_to_mat, underlying_price, strike_price, stock_price_hist: list, interest_rate):
        super().__init__(time_to_mat, underlying_price, strike_price, stock_price_hist)
        self.interest_rate = interest_rate
        self.calculate = self.calculate_both
        self.price_call = None
        self.price_put = None
        # Not using static methods here, because it is not necessary to calculate both call and put prices
        # upon instance creation - only one of them might be needed.

    def calc_d(self):
        d1 = ( 1 / (self.sigma*numpy.sqrt(self.time_to_mat)) ) * \
             (numpy.log(self.underlying_price/self.strike_price) +
              (self.interest_rate + (self.sigma**2)/2 )*self.time_to_mat)
        d2 = d1 - self.sigma*numpy.sqrt(self.time_to_mat)
        return d1, d2

    def calc_price_call(self):
        logging.info(f'Initiating calculation of Call option price using B-S method, with the following parameters: \
Time to maturity = {self.time_to_mat}, Underlying price = {self.underlying_price}, Strike price = {self.strike_price}, \
Interest rate = {self.interest_rate}, Volatility = {self.sigma}')
        N = norm.cdf    # Cumulative Distribution Function
        d1, d2 = self.calc_d()
        logging.debug(f'B-S calculated values: d1={d1}, d2={d2}')
        # Black-Scholes formula explanation:
        # Simplified:
        # price_call =  Return - Cost
        # price_call = (Stock Price * Probability Func) - (Probability Func X Strike Price X Discount to Present Value)
        # for Probability Func we take into account volatility (sigma).
        # Lower volatility would result in a narrower bell curve (higher certainty of price).
        ret = N(d1) * self.underlying_price
        cost = N(d2) * self.strike_price * numpy.exp(-self.interest_rate * self.time_to_mat)
        self.price_call = ret - cost
        logging.info(f'B-S calculated Call option price = {self.price_call}')

    def calc_price_put(self):
        logging.info(f'Initiating calculation of Put option price using B-S method, with the following parameters: \
Time to maturity = {self.time_to_mat}, Underlying price = {self.underlying_price}, Strike price = {self.strike_price}, \
Interest rate = {self.interest_rate}, Volatility = {self.sigma}')
        N = norm.cdf    # Cumulative Distribution Function
        d1, d2 = self.calc_d()
        logging.debug(f'B-S calculated values: d1={d1}, d2={d2}')
        # The put option formula is derived from the
        # "put-call parity" expression, which is outside the scope of this project
        # The resulting formula is:
        self.price_put = N(-d2) * self.strike_price * numpy.exp(-self.interest_rate * self.time_to_mat) \
                    - N(-d1) * self.underlying_price
        logging.info(f'B-S calculated Call option price = {self.price_put}')

    def calculate_both(self):
        """for the purposes of the plot() function in the super class."""
        self.calc_price_call()
        self.calc_price_put()
        return self.price_call, self.price_put


class MonteCarlo(Pricer):
    # There's conflicting info on what Monte Carlo actually is.
    # From this https://www.wallstreetmojo.com/option-pricing-2/:
    # Monte Carlo is a method of calculating option prices by taking random
    # variables (unlike BS model that takes 5 give variables).
    # The Monte Carlo option model is the application of Monte Carlo Methods.
    # This pricing model uses random samples to calculate the price.
    # This method is more favorable than other methods like Black-Scholes
    # for calculating the value of options with multiple sources of uncertainty.
    # From wikipedia: Since the underlying random process is the same,
    # for enough price paths, the value of a european option here should be
    # the same as under Black–Scholes.
    # MC process:
    # 1. Simulate many stock price trajectories
    # 2. Average the stock price for each of the simulated paths.
    # 3. Calculate the payoff (option premium) for each simulated price.
    # 4. Average the payoffs for all paths.
    # http://www.goddardconsulting.ca/option-pricing-monte-carlo-index.html

    # Weiner process (Used by MC to simulate stock price evolution):
    # price_t   -> underlying price at time t
    # price_0 = 175   # underlying price now - in USD
    # mu = 0.05   # asset's historical return TODO: find a formula for this
    # volat = calc_volatility_log(apple_price_hist_2)     # expected volatility
    # dt = 1/252  # time step - in years
    # z = numpy.random.normal()   # a random number from a normal distribution
    # with a peak of 0, 1std distribution is +/- 1, etc.

    # price_t = price_0*numpy.exp(((mu-(volat**2)/2)*dt)+(volat*numpy.sqrt(dt)*z))

    def __init__(self, time_to_mat, underlying_price, strike_price, stock_price_hist: list, mu, n):
        super().__init__(time_to_mat, underlying_price, strike_price, stock_price_hist)
        self.mu = mu
        self.n = n
        self.calculate = self._calc_for_plot
        self.price_hist = None
        self.sim_price = None
        self.call_price = None
        self.put_price = None
        # Not using static method for price calculation to be in line with BS method

    def calc_mc(self):
        # Weiner process (Used by MC to simulate stock price evolution):
        # price_t   -> underlying price at time t
        # price_0   -> starting underlying price - in USD
        # mu     -> asset's historical return TODO: find a formula for this
        # volat     -> expected volatility
        # steps    -> time step - in work days
        # z     -> a random number from a normal distribution
        # with a peak of 0, 1std distribution is +/- 1, etc.
        # TODO: calculate only last value and move trajectory calc to another def (for creating a plot later)
        logging.info(f'Initiating stock price simulation using the Monte Carlo model, with the following parameters: \
Time to maturity = {self.time_to_mat}, Underlying price at start = {self.underlying_price}, \
Asset\'s historical return = {self.mu}, Volatility = {self.sigma}')
        # pre-calculating all z values in the following array
        # simple testing showed reduction of ~10% in processing time compared to
        # calculating z value every iteration.
        steps = round(self.time_to_mat * 252)  # turning years into work days
        z_array = numpy.random.normal(size=steps)
        dt = self.time_to_mat/steps
        price_t = self.underlying_price*numpy.exp(((self.mu-(self.sigma**2)/2)*dt)+(self.sigma*numpy.sqrt(dt)*z_array[0]))
        self.price_hist = [price_t]
        for i in range(steps):
            z = z_array[i]
            price_t = price_t*numpy.exp(((self.mu-(self.sigma**2)/2)*dt)+(self.sigma*numpy.sqrt(dt)*z))
            self.price_hist.append(price_t)
        self.sim_price = self.price_hist[-1]
        logging.info(f'Calculated stock price after {self.time_to_mat} years: {self.sim_price}')

    # Call option payoff is calculated as: Spot price - Strike price
    # Put option payoff is calculated as: Strike price - Spot price
    # Payoff cannot be negative.

    def option_price_mc(self):
        """Calculate option price for given parameters
        n - number of iterations for simulation to run
        Simulate n trajectories and calculate payoff for each.
        Return averaged payoff"""
        logging.debug(f'Starting Option price calculation from MC simulated prices with {self.n} simulations')
        # Initializing empty numpy arrays:
        simulated_strike_prices = numpy.empty(self.n)
        simulated_call_payoffs = numpy.empty(self.n)
        simulated_put_payoffs = numpy.empty(self.n)

        for i in range(self.n):
            self.calc_mc()
            simulated_strike_prices[i] = self.sim_price
            # Currently not used. Keeping for future use for plots
            simulated_call_payoffs[i] = max(self.sim_price - self.strike_price, 0)
            simulated_put_payoffs[i] = max(self.strike_price - self.sim_price, 0)

        self.call_price = numpy.mean(simulated_call_payoffs)
        self.put_price = numpy.mean(simulated_put_payoffs)
        logging.info(f'Calculated option prices using MC method: Call = {self.call_price}, Put = {self.put_price}')

    def _calc_for_plot(self):
        # This function is different than calculate_both in the B-S class, because
        # the MC calc functions are more complex and cannot simply be fed 2d arrays.
        # Thus, the 2d arrays are iterated over and each value is individually fed
        # to the calc function above. Then a new 2D array containing the option prices
        # is constructed.
        strike_price_array = self.strike_price
        time_to_mat_array = self.time_to_mat
        call_price_2d_array = numpy.empty([len(strike_price_array),len(time_to_mat_array[0])])
        put_price_2d_array = numpy.empty([len(strike_price_array),len(time_to_mat_array[0])])

        for i in range(len(time_to_mat_array)):
            self.time_to_mat = time_to_mat_array[i][0]    # assuming all elements in inner arrays are the same
            for j in range(len(strike_price_array[0])):
                self.strike_price = strike_price_array[0][j]
                self.option_price_mc()
                call_price_2d_array[i][j] = self.call_price
                put_price_2d_array[i][j] = self.put_price
        return call_price_2d_array, put_price_2d_array
