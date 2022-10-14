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

print('# Computing optimal weights')

print(f'Solution for a) is:\n{Wa}')
print(f'Solution for b) is:\n{Wb}')

# 4. Test with train inputs
print(f'Predictions for model a) are:\n{np.dot(Xa, Wa)}')
print(f'Predictions for model b) are:\n{np.dot(Xb, Wb)}\n')

# 5. Test uniqueness of solution
A, B = np.dot(Xa.T, Xa), np.dot(Xb.T, Xb)
detA, detB = np.linalg.det(A), np.linalg.det(B)
print('# Testing solution uniqueness')

## 5.1. If determinant of A''A is > 0
print('## Determinants A''A')
print(f"Determinant of A''A for model a): {detA}")
print(f"Determinant of A''A for model b): {detB}")

## 5.2. If singular values matrix has at least one singular value != 0
ua, sa, vha = np.linalg.svd(A)
ub, sb, vhb = np.linalg.svd(B)
Sa, Sb = np.diag(sa), np.diag(sb)
print('## Singular value decomposition')
print(f'Sa is:\n{Sa}')
print(f'Sb is:\n{Sb}')
rank_Sa = np.linalg.matrix_rank(Sa)
rank_Sb = np.linalg.matrix_rank(Sb)

if rank_Sa < len(sa):
	print(f'>> Model a) has not an unique solution because rank of Sa {rank_Sa} < len(sa) {len(sa)}')
if rank_Sb < len(sb):
	print(f'>> Model b) has not an unique solution because rank of Sb {rank_Sb} < len(sb) {len(sb)}')

