package com.vonhessling.neuralnetwork;

public class MathUtils {

	
	/**
	 * Sigmoid function
	 * @param x The input value
	 * @return The sigmoid of the input value
	 */
	public static double sigmoid(double x) {
		return (1 / ( 1 + Math.pow(Math.E,(-1 * x))));
	}
	
	/**
	 * Derivative of sigmoid function.
	 * @param x The input value
	 * @return The sigmoid derivative of the input value
	 */
	public static double sigmoidDerivative(double x) {
		double s = sigmoid(x);
		return s * (1-s);		
	}	

}
