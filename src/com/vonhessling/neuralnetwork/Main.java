package com.vonhessling.neuralnetwork;
import java.io.IOException;

import org.apache.log4j.Logger;


public class Main {
    private static final Logger log = Logger.getLogger(NeuralNetwork.class);
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		int[] numNeurons = {7,3,1};
		NeuralNetwork nn = new NeuralNetwork(numNeurons);
		try {
			nn.loadExamples("var/abaloneNormalized.csv", Mode.TRAINING);
		} catch (IOException e) {
			log.error(e);
		}
		nn.trainExamples(1000, 0.25);
		nn.printWeights();
		nn.printNeurons();

	}

}
