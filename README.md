# Options pricer

Options pricer written on Python. It employs two different methods of calculating an option price: Black-Scholes and Monte Carlo option model.

### Black-Scholes
It employs the Black-Scholes formula to calculate the option price based on the following input parameters: 
- Time to maturity
- Underlying asset price
- Strike price
- Stock price history
- Interest rate
This model uses a fixed formula with no degree of randomness. It will always return the same price given the same initial conditions.


### Monte Carlo option model
This method simulates what the stock price is expected to be at the time of maturity and it uses that to calculate what the option price should be. Process:
    1. Simulate many stock price trajectories
    2. Average the stock price for each of the simulated paths.
    3. Calculate the payoff (option premium) for each simulated price.
    4. Average the payoffs for all paths.
It takes almost the same parameters as the Black-Scholes model plus the asset's historical return (mu).
    

## Usage

Run main.py to calculate option price for sample parameters for a sample stock (Apple) and generate a 3D surface for each. Code can be simply edited to change the parameters.

## Features

- Abiltiy to calculate call and put option prices with custom parameters (given above), using either Black-Scholes model or the Monte Carlo option model.
- Ability to generate a 3D surface of the option price vs strike price vs time to maturity, e.g.

![image](https://github.com/istbozhkov/Pricer/assets/113388063/934adefa-9ed4-49c4-ba47-1b44d343334a)
![image](https://github.com/istbozhkov/Pricer/assets/113388063/7813e752-1fde-4797-9ec9-3deec3cc877f)

