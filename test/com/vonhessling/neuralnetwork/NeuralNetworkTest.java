package com.vonhessling.neuralnetwork;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class NeuralNetworkTest {
	int[] numNeurons = {7,3,1};
	NeuralNetwork nn;
	
	@Before
	public void setUp() throws Exception {
		 nn = new NeuralNetwork(numNeurons);
	}

	@After
	public void tearDown() throws Exception {
	}

	@Test
	public void testSigmoid() {
		assertTrue(new Double(0.9999546021312976).equals(nn.sigmoid(10)));
		assertTrue(new Double(0.5).equals(nn.sigmoid(0)));
		assertTrue(new Double(0.7310585786300049).equals(nn.sigmoid(1)));
		assertTrue(new Double(0.11920292202211757).equals(nn.sigmoid(-2)));
		
	}
	@Test
	public void testSigmoidDerivative() {
		assertTrue(new Double(0.25).equals(nn.sigmoidDerivative(0)));
		assertTrue(new Double(0.19661193324148185).equals(nn.sigmoidDerivative(1)));
		assertTrue(new Double(0.10499358540350653).equals(nn.sigmoidDerivative(-2)));		
	}

}
