import pricer_classes

# Sample price history
price_hist = [150, 153.07, 154.94, 155.86, 156.82, 153.3, 159.92,
              152.70, 157.76, 156.30, 154.68, 165.25, 166.44, 166.52,
              171.25, 174.18, 174.24, 170.94, 170.41, 169.43, 170.89, 170.18]

apple_price_hist = [142.449997, 146.100006, 146.399994, 145.429993, 140.089996, 140.419998, 138.979996, 138.339996,
                    142.990005, 138.380005, 142.410004, 143.75, 143.860001, 143.389999, 147.270004, 149.449997,
                    152.339996, 149.350006, 144.800003, 155.740005, 153.339996, 150.649994, 145.029999, 138.880005,
                    138.380005, 138.919998, 139.5, 134.869995, 146.869995, 149.699997, 148.279999, 150.039993,
                    148.789993, 150.720001, 151.289993, 148.009995, 150.179993, 151.070007, 148.110001, 144.220001,
                    141.169998, 148.029999, 148.309998, 147.809998, 146.630005, 142.910004, 140.940002, 142.649994,
                    142.160004, 144.490005, 145.470001, 143.210007, 136.5, 134.509995, 132.369995, 132.300003,
                    135.449997, 132.229996, 131.860001, 130.029999, 126.040001, 129.610001, 129.929993, 125.07,
                    126.360001, 125.019997, 129.619995, 130.149994, 130.729996, 133.490005, 133.410004, 134.759995,
                    135.940002, 135.210007, 135.270004, 137.869995, 141.110001, 142.529999, 141.860001, 143.960007,
                    145.929993, 143, 144.289993, 145.429993, 150.820007, 154.5, 151.729996, 154.649994, 151.919998,
                    150.869995, 151.009995, 153.850006, 153.199997, 155.330002, 153.710007, 152.550003, 148.479996,
                    148.910004, 149.399994, 146.710007, 147.919998, 147.410004, 145.309998, 145.910004, 151.029999,
                    153.830002, 151.600006, 152.869995, 150.589996, 148.5, 150.470001, 152.589996, 152.990005,
                    155.850006, 155, 157.399994, 159.279999, 157.830002, 158.929993, 160.25, 158.279999, 157.649994,
                    160.770004, 162.360001, 164.899994, 166.169998, 165.630005, 163.759995, 164.660004, 162.029999,
                    160.800003, 160.100006, 165.559998, 165.210007, 165.229996, 166.470001, 167.630005, 166.649994,
                    165.020004, 165.330002, 163.770004, 163.759995, 168.410004, 169.679993, 169.589996, 168.539993,
                    167.449997, 165.789993, 173.570007, 173.5, 171.770004, 173.559998, 173.75, 172.570007, 172.070007,
                    172.070007, 172.690002, 175.050003, 175.160004, 174.199997, 171.559998, 171.839996, 172.990005,
                    175.429993, 177.300003, 177.25, 180.089996, 180.949997, 179.580002, 179.210007, 177.820007,
                    180.570007, 180.960007, 183.789993, 183.309998, 183.949997, 186.009995, 184.919998, 185.009995,
                    183.960007, 187, 186.679993, 185.270004, 188.059998, 189.25, 189.589996, 193.970001, 192.460007,
                    191.330002, 191.809998, 190.679993, 188.610001, 188.080002, 189.770004, 190.539993, 190.690002,
                    193.990005, 193.729996, 195.100006, 193.130005, 191.940002, 192.75, 193.619995, 194.5, 193.220001,
                    195.830002, 196.449997, 195.610001, 192.580002, 191.169998, 181.990005, 178.850006, 179.800003,
                    178.190002, 177.970001, 177.789993, 179.460007, 177.449997, 176.570007, 174, 174.490005, 175.839996,
                    177.229996, 181.119995, 176.380005, 178.610001, 180.190002, 184.119995, 187.649994, 187.869995,
                    189.460007, 189.699997, 182.910004, 177.559998, 178.179993, 179.360001, 176.300003, 174.210007,
                    175.740005, 175.009995, 177.970001, 179.070007, 175.490005, 173.929993, 174.789993, 176.080002,
                    171.960007, 170.429993, 170.690002, 171.210007, 173.75, 172.40, 173.66, 175.01
                    ]

apple_price_hist_2 = [177.970001, 179.070007, 175.490005, 173.929993, 174.789993,
                      176.080002, 171.960007, 170.429993, 170.690002, 171.210007,
                      173.75, 172.40, 173.66, 175.01]


# Sample values for B-S:
t = 1     # in years
S = 174.9   # Underlying price USD
K = 190     # Strike price USD
r = 0.05
# sig = calc_volatility_log(apple_price_hist_2)

apple_BS = pricer_classes.BlackScholes(time_to_mat=t, underlying_price=S, strike_price=K,
                                       stock_price_hist=apple_price_hist, interest_rate=r)

for K in range(140,191,5):
    apple_BS.strike_price = K
    apple_BS.calc_price_call()
    apple_BS.calc_price_put()
    print(f"K={K}, call: S={apple_BS.price_call}, put: S={apple_BS.price_put}")

# Sample values for MC:
p0 = 175
mu = 0.05
step = 252
t = 1
n = 100

apple_MC = pricer_classes.MonteCarlo(time_to_mat=t, underlying_price=p0, strike_price=180,
                                     stock_price_hist=apple_price_hist, mu=mu)

apple_MC.option_price_mc(n=n)
print(f"MC calculated prices with {n} iterations: call = {apple_MC.call_price}, put = {apple_MC.put_price}")

