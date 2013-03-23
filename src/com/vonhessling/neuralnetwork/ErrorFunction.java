package com.vonhessling.neuralnetwork;

public interface ErrorFunction {
	public double getError(double input);
	public double getErrorDerivative(double input);
}
