package com.vonhessling.neuralnetwork;

import static org.junit.Assert.*;

import java.io.IOException;

import org.apache.log4j.Logger;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class NeuralNetworkTest {
    private static final Logger log = Logger.getLogger(NeuralNetworkTest.class);
	
	@Before
	public void setUp() throws Exception {
	}

	@After
	public void tearDown() throws Exception {
	}

	@Test
	public void testSigmoid() {
		assertTrue(new Double(0.9999546021312976).equals(NeuralNetwork.sigmoid(10)));
		assertTrue(new Double(0.5).equals(NeuralNetwork.sigmoid(0)));
		assertTrue(new Double(0.7310585786300049).equals(NeuralNetwork.sigmoid(1)));
		assertTrue(new Double(0.11920292202211757).equals(NeuralNetwork.sigmoid(-2)));
		
	}
	@Test
	public void testSigmoidDerivative() {
		assertTrue(new Double(0.25).equals(NeuralNetwork.sigmoidDerivative(0)));
		assertTrue(new Double(0.19661193324148185).equals(NeuralNetwork.sigmoidDerivative(1)));
		assertTrue(new Double(0.10499358540350653).equals(NeuralNetwork.sigmoidDerivative(-2)));		
	}
	
	@Test
	public void testTrainingExampleLoading() {
		int[] numNeurons = {7,3,1};
		NeuralNetwork nn = new NeuralNetwork(numNeurons);
		
		try {
			nn.loadExamples("var/abaloneNormalized.csv", Mode.TRAINING);
		} catch (IOException e) {
			fail("Can't load training file with examples!");
		}
		assertTrue(nn.getNumExamples(Mode.TRAINING) == 4177);
	}
	
	@Test
	public void testTestingExampleLoading() {
		int[] numNeurons = {7,3,1};
		NeuralNetwork nn = new NeuralNetwork(numNeurons);
		
		try {
			nn.loadExamples("var/abaloneNormalizedTesting.csv", Mode.TESTING);
		} catch (IOException e) {
			fail("Can't load testing file with examples!");
		}
		assertTrue(nn.getNumExamples(Mode.TESTING) == 2);
	}

	
	@Test
	public void testFullCycle() {
		int[] numNeurons = {7,3,1};
		NeuralNetwork nn = new NeuralNetwork(numNeurons);
		
		try {
			nn.loadExamples("var/abaloneNormalized.csv", Mode.TRAINING);
			nn.loadExamples("var/abaloneNormalizedTesting.csv", Mode.TESTING);
		} catch (IOException e) {
			fail("Can't load training/testing file with examples!");
		}
		nn.trainExamples(1000, 0.25);
		nn.testExamples();
	}

	@Test
	public void testPrediction() {
		int[] numNeurons = {7,3,1};
		NeuralNetwork nn = new NeuralNetwork(numNeurons);
		
		try {
			nn.loadExamples("var/abaloneNormalized.csv", Mode.TRAINING);
		} catch (IOException e) {
			fail("Can't load training file with examples!");
		}
		nn.trainExamples(1000, 0.25);
		
		double[] input = {0.585,0.455,0.17,0.9945,0.4255,0.263,0.2845};
		double[] output = {0.366666667};
		Example example = new Example(input, output);
		nn.predict(example);
		log.info("Prediction: " + example);
		log.info("Real output was " + output[0]);
		assertTrue(example.getOutput()[0] > 0);
	}

}
