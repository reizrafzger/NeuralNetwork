package com.vonhessling.neuralnetwork;

public class SigmoidErrorFunction implements ErrorFunction {

	public double getError(double input) {
		return MathUtils.sigmoid(input);
	}
	
	public double getErrorDerivative(double input) {
		return MathUtils.sigmoidDerivative(input);
	}

}
