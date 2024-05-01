import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from decimal import Decimal
import sklearn as sk
from sklearn.metrics import  mean_absolute_error, mean_absolute_percentage_error

np.random.seed(42)
PATH_TO_NODE_FEATURES = "node_features.txt"
fullData = pd.read_csv(filepath_or_buffer=PATH_TO_NODE_FEATURES, sep=",")
fullData = fullData[fullData['dateTime'] > '2021-04-06']


fullData['dateTime'] = pd.to_datetime(fullData['dateTime'], format='%Y-%m-%d')


elevation = fullData['altitude']
noise = np.random.normal(0, 0.1, size = fullData['altitude'].shape)
noisy_elevation = elevation+noise

fullData['altitude'] = fullData['altitude'].apply(lambda x: round(x, 10))
fullData['altitude'] = noisy_elevation



pivoted_data = fullData.pivot_table(index='dateTime', columns='site_no', values=['nitrite+nitrate', 'altitude'], aggfunc='mean')
pivoted_data.dropna(inplace=True) 



pivoted_data_2 = fullData.pivot_table(index='dateTime', columns='site_no', values=['nitrite+nitrate'], aggfunc='mean')
pivoted_data_2.dropna(inplace=True)


two_d_arr_1 = np.array((pivoted_data.values))
two_d_arr_2 = np.array(pivoted_data_2.values)

#pivoted_data = pivoted_data.astype(float)
pivoted_data_1 = pd.DataFrame(pivoted_data.values[0:340])

pivoted_data_2 = pivoted_data_2.astype(float)
pivoted_data_2 = pd.DataFrame(pivoted_data_2.values[0:340])




test_conc = pivoted_data_1.values[-19:, 24:]
print(test_conc)
print(test_conc.shape)
for i in range(10):

    print("Test conc shape is")
    print(test_conc.shape)
    model = VAR(pivoted_data_1)


    #pivoted_data.reset_index(drop=True, inplace=True)

    results = model.fit(maxlags= 7)
    lag_order = results.k_ar
    forecast_values = results.forecast(pivoted_data_1.values[-lag_order:], 19)

    forecast_values = forecast_values[:, 24:]
    ##print(forecast_values[23:])
    mape = mean_absolute_percentage_error(test_conc, forecast_values)
    mae = mean_absolute_error(test_conc,forecast_values )
    print(mape)
    print(mae)
