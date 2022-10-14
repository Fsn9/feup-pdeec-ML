import numpy as np

# 1. Saving data
x1 = np.array([340, 665, 368, 331, 954]).reshape(-1, 1)
x2 = np.array([16, 25, 15, 15, 40]).reshape(-1, 1)
x3 = np.array([356, 690, 383, 346, 994]).reshape(-1, 1)
Y = np.array([1.5, 2.8, 1.7, 1.3, 5.0]).reshape(-1, 1)

# 2. Make inputs X for both a) and b)
Xa = np.concatenate((x1,x2), axis = 1)
Xb = np.concatenate((x1,x2,x3), axis = 1)

# 3. Find optimal weights: w* = (X'X)^(-1) * X' * Y
# a) and b)
Wa = np.dot(np.dot(np.linalg.inv(np.dot(Xa.T, Xa)), Xa.T), Y)
Wb = np.dot(np.dot(np.linalg.inv(np.dot(Xb.T, Xb)), Xb.T), Y)

print(f'Solution for a) is: {Wa}')
print(f'Solution for b) is: {Wb}')

# 4. Test with train inputs
Xa_test = np.array([331,15])
Xb_test = np.array([331,15,346])
y_pred_a = np.dot(Wa.T, Xa_test)
y_pred_b = np.dot(Wb.T, Xb_test)

print(f'Predictions for model a) are: {np.dot(Xa, Wa)}')
print(f'Predictions for model b) are: {np.dot(Xb, Wb)}')
