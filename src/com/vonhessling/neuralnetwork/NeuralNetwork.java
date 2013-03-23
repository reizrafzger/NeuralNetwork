package com.vonhessling.neuralnetwork;

import java.io.IOException;

public interface NeuralNetwork {
	public void loadExamples(String fileName, Mode mode) throws IOException;
	public void trainExamples(int numTimes, double learningRate);
	public void setErrorFunction(ErrorFunction errorFunction);
	public void testExamples();
	public void predict(Example example);


}
