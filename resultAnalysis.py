import numpy as np

# Results for 5 runs of STGNN with elevation
mape = [7.354521736266121, 7.57837156265501, 7.633404928540426, 7.357801649305555, 7.64471677749876]
mae = [0.29349893236917163, 0.30156078035869294, 0.30447030445886036, 0.29379414755200584, 0.3035054888044085]
print("With elevation:")
std_dev = np.std(mape, ddof=1) 
standard_error = std_dev / np.sqrt(len(mape))
print("MAPE: ", np.mean(mape), " +- " , standard_error)

std_dev = np.std(mae, ddof=1)  
standard_error = std_dev / np.sqrt(len(mae))
print("MAE: ", np.mean(mae), " +- " , standard_error)

# Results for 5 runs of STGNN without elevation
mape = [9.174015105716766, 8.435066344246032, 9.415150475880456, 10.11674572172619, 11.920594230530755]
mae = [0.34557342529296875, 0.32527732849121094, 0.35953885033017113, 0.37630044846307664, 0.4287273467533172]
print("WithOUT elevation:")
std_dev = np.std(mape, ddof=1) 
standard_error = std_dev / np.sqrt(len(mape))
print("MAPE: ", np.mean(mape), " +- " , standard_error)

std_dev = np.std(mae, ddof=1)  
standard_error = std_dev / np.sqrt(len(mae))
print("MAE: ", np.mean(mae), " +- " , standard_error)
