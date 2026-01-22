import numpy as np
from itertools import combinations_with_replacement

from dataset_genarator import generator
from PCA_tranformer import pca_tranformer


def polynomial_features(x, degree):
	x = np.asarray(x)
	if x.ndim == 1:
		x = x.reshape(-1, 1)
	features = [np.ones(x.shape[0])]
	for d in range(1, degree + 1):
		for combo in combinations_with_replacement(range(x.shape[1]), d):
			term = np.prod(x[:, combo], axis=1)
			features.append(term)
	return np.column_stack(features)


def fit_polynomial_regression(x_t, y, degree=3):
	design = polynomial_features(x_t, degree)
	targets = np.asarray(y).reshape(-1)
	coeffs, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
	return coeffs


def predict_polynomial_regression(x_t, coeffs, degree=3):
	design = polynomial_features(x_t, degree)
	return design @ coeffs


def compute_regression_metrics(preds, y):
	targets = np.asarray(y).reshape(-1)
	residuals = preds - targets
	mse = float(np.mean(residuals ** 2))
	rmse = float(np.sqrt(mse))
	y_mean = float(np.mean(targets))
	ss_tot = float(np.sum((targets - y_mean) ** 2))
	ss_res = float(np.sum(residuals ** 2))
	r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
	return {"mse": mse, "rmse": rmse, "r2": r2}


def train_model(n_samples=100000, degree=2):
	X, y = generator(n_samples)
	x_t, eigval, compo = pca_tranformer(X)
	coeffs = fit_polynomial_regression(x_t, y, degree)
	preds = predict_polynomial_regression(x_t, coeffs, degree)
	metrics = compute_regression_metrics(preds, y)
	return coeffs, metrics, x_t, y, eigval, compo


def main():
	degree = 2
	coeffs, metrics, x_t, y, eigval, compo = train_model(degree=degree)
	preds = predict_polynomial_regression(x_t, coeffs, degree)
	print(f"Training samples: {len(y)}")
	print(f"Polynomial degree: {degree}")
	print(f"MSE: {metrics['mse']:.4f}")
	print(f"RMSE: {metrics['rmse']:.4f}")
	print(f"R2 (accuracy): {metrics['r2']:.4f}")
	print("First five predictions vs targets:")
	for pred, target in zip(preds[:5], y[:5]):
		print(f"pred: {pred:.3f} target: {target:.3f}")


if __name__ == "__main__":
	main()
