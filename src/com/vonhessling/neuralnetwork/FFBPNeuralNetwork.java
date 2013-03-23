package com.vonhessling.neuralnetwork;
import java.io.File;
import java.io.IOException;
import java.util.*;
import org.apache.log4j.Logger;
import org.apache.commons.io.FileUtils;

/**
 * Implementation of a feed-forward backpropagation neural network
 * 
 * @TODO Could add momentum
 * 
 * @author vonhessling
 *
 */
public class FFBPNeuralNetwork implements NeuralNetwork {
    private static final Logger log = Logger.getLogger(FFBPNeuralNetwork.class);
	
	// [layerIndex][fromNeuron][toNeuron]
	private double[][][] weights; 

	// [layerIndex][neuronIndex]
	private Neuron[][] neurons;
	
	private List<Example> trainingExamples;

	private List<Example> testingExamples;
	
	// default error function is sigmoid
	private ErrorFunction errorFunction = new SigmoidErrorFunction();  
	
	/**
	 * Creates a new neural network. Initializes weights randomly
	 * @param learningRate
	 *            The learning rate
	 * @param numNeurons
	 *            Describes the number of neurons per layer; Thus also defines
	 *            the number of layers
	 */
	public FFBPNeuralNetwork(int[] numNeurons) {
		this.trainingExamples = new ArrayList<Example>();
		
		// initialize neurons in all layers
		neurons = new Neuron[numNeurons.length][];
		for (int layer = 0; layer < numNeurons.length; layer++) {
			neurons[layer] = new Neuron[numNeurons[layer]];
			for (int neuronIndex = 0; 
					neuronIndex < neurons[layer].length; 
					neuronIndex++) {
				neurons[layer][neuronIndex] = new Neuron();
			}
		}
		
		// initialize weights
		// there is one less weight than layers
		weights = new double[numNeurons.length - 1][][]; 
		for (int layer = 0; layer < numNeurons.length - 1; layer++) {
			//For the from layer, add + 1 for bias weight
			weights[layer] = new double[numNeurons[layer] + 1]  // "from" 
										[numNeurons[layer + 1]];// "to"
			for (int from = 0; from < weights[layer].length; from++) {
				for (int to = 0; to < weights[layer][from].length; to++) {
					if (from == weights[layer].length - 1) {
						// bias weight in the last from column:
						weights[layer][from][to] = -1; 
					} else {
						// initialize regular weights randomly:
						weights[layer][from][to] = Math.random(); 
					}
				}
			}
		}
	}

	/**
	 * Loads the examples in the given file
	 * 
	 * @param fileName The name of the file to load
	 * @param mode Testing/Training mode
	 * @throws IOException Throws IOException if I/O error occurs with file
	 */
	public void loadExamples(String fileName, Mode mode) throws IOException {
		log.info("Loading examples from " + fileName + " for " + mode + "...");
		int size = 0;
		switch (mode) {
			case TRAINING: {
				trainingExamples = loadExamples(fileName);
				size = trainingExamples.size();
			}
			case TESTING: {
				testingExamples = loadExamples(fileName);
				size = testingExamples.size();
			}
		}
		log.info(size + " examples loaded.");
	}

	
	/**
	 * Loads all examples from the desired file.
	 * 
	 * @param fileName
	 *            The name of the file to load
	 * @return Returns the examples
	 * @throws IOException
	 *             Throws IOException if I/O error occurs with file or if file
	 *             is malformatted
	 */
	private List<Example> loadExamples(String fileName) throws IOException {
		List<Example> loadedExamples = new ArrayList<Example>();
		
		List<String> lines = FileUtils.readLines(new File(fileName));
		for (String line : lines) {
			// # of neurons on layer 0
			double[] inputVector = new double[neurons[0].length]; 
			// # of output neurons: 
			double[] outputVector = 
					new double[neurons[neurons.length - 1].length];
			
			String[] values = line.split(",");
			if (values.length != inputVector.length + outputVector.length) {
				throw new IOException("Line contains " + values.length
						+ " items, not "
						+ (inputVector.length + outputVector.length)
						+ " as required. Line content: " + line);
			}
			int i = 0;
			int o = 0;
			for (String value : values) {
				if (i < inputVector.length) {
					inputVector[i++] = Double.parseDouble(value.trim());
				} else {
					outputVector[o++] = Double.parseDouble(value.trim());
				}
			}
			loadedExamples.add(new Example(inputVector, outputVector));			
		}		
		return loadedExamples;
	}
	
	
	/**
	 * Trains the neural network using all loaded examples numTimes times
	 * 
	 * @param numTimes
	 *            The number of times the examples are passed into the network
	 * @param learningRate
	 *            The learning rate parameter for the neural network
	 */	
	public void trainExamples(int numTimes, double learningRate) {
		if (trainingExamples == null) {
			throw new IllegalStateException(
					"Need to load training examples first!");
		}
		log.info("Training Neural Network " + numTimes + " times with "
				+ trainingExamples.size() + " examples...");
		for (int i = 0; i < numTimes; i++) {
			trainExamples(learningRate);
		}		
	}
	
	/**
	 * Trains the neural network using all loaded examples a single time.
	 * @param learningRate The learning rate to use.
	 */
	private void trainExamples(double learningRate) {
		for (Example example : trainingExamples) {
			// load values...
			loadExampleInInputLayer(example);
			// propagate values forward...
			propagateForward();

			// compute error in the output layer...	
			Neuron currentNeuron;
			// for the output layer only, compute the error:
			int outputLayer = neurons.length - 1;  
			// update each neuron's input value:
			for (int i = 0; i < neurons[outputLayer].length; i++) {
				currentNeuron = neurons[outputLayer][i];
				currentNeuron.setError(
						computeOutputError(
								currentNeuron.getInput(), 
								currentNeuron.getOutput(), 
								example.getOutput()[i]));				
			}


			// propagate error back layer by layer and 
			// update weights accordingly:
			
			// for each layer, backwards
			for (int layer = neurons.length - 2; layer >= 0; layer--) { 
				for (int j = 0; j < neurons[layer].length; j++) {
					currentNeuron = neurons[layer][j];
					// update error in current layer:
					currentNeuron.setError(errorFunction.getErrorDerivative(
							currentNeuron.getInput())
							* getTotalOutputError(layer, j));
					// and finally, update the weights:
					for (int to = 0; to < weights[layer][j].length; to++) {
						weights[layer][j][to] = weights[layer][j][to] + 
								learningRate * 
								currentNeuron.getOutput() * 
								neurons[layer+1][to].getError();
					}
				}
			}	
		}
	}
	
	/**
	 * Sets the error function
	 * @param errorFunction The error function to use
	 */
	public void setErrorFunction(ErrorFunction errorFunction) {
		this.errorFunction = errorFunction;
	}
	
	/**
	 * Propagates values forward through the neural network up to the output
	 * layer
	 */
	private void propagateForward() {
		Neuron currentNeuron;
		// for each layer...
		for (int layer = 1; layer < neurons.length; layer++) {
			// update each neuron's input value:
			for (int i = 0; i < neurons[layer].length; i++) {  
				currentNeuron = neurons[layer][i];
				// find input value by summing up the output of the previous
				// layer, weighted
				currentNeuron.setInput(getInputStrengthForNeuron(layer, i));
				// this neuron's output is the sigmoid value of the input
				currentNeuron.setOutput(errorFunction.getError(currentNeuron
						.getInput()));
			}
		}		
	}
	
	/**
	 * Tests all testingExamples and prints predicted output values
	 */
	public void testExamples() {
		if (testingExamples == null) {
			throw new IllegalStateException(
					"Need to load testing examples before " +
					"attempting to predict!");
		}
		for (Example example : testingExamples) {
			log.info("Testing: " + example);
			loadExampleInInputLayer(example);
			propagateForward();
			int outputLayer = neurons.length - 1;
			log.info("Prediction: ");
			for (int i = 0; i < neurons[outputLayer].length; i++) {
				log.info(neurons[outputLayer][i].getOutput());
			}
		}
	}
	
	/**
	 * Predicts the output for this example and updates it
	 * @param example The example to find the prediction for
	 */
	public void predict(Example example) {
		loadExampleInInputLayer(example);
		propagateForward();
		int outputLayer = neurons.length - 1;
		double[] output = new double[neurons[outputLayer].length];
		for (int i = 0; i < neurons[outputLayer].length; i++) {
			// just copy the output of the neural network to the example:
			output[i] = neurons[outputLayer][i].getOutput();
		}
		example.setOutput(output);		
	}
	
	/**
	 * Loads the given example into the input layer of the network
	 * @param example The example to load
	 */
	private void loadExampleInInputLayer(Example example) {
		// set input-neuron's output value to the example's input vector
		for (int j = 0; j < neurons[0].length; j++) { 
			neurons[0][j].setOutput(example.getInput()[j]);
		}		
	}
	
	/**
	 * Returns the input strength of the given neuron in the given layer by 
	 * looking up the previous layer's output and the weights
	 * @param layer The layer of the neuron
	 * @param neuronIndex The index of the neuron
	 * @return The input strength for the given neuron
	 */
	private double getInputStrengthForNeuron(int layer, int neuronIndex) {
		double sum = 0;
		double output; 
		double weight;
		for (int i = 0; i < neurons[layer-1].length; i++) {
			output = neurons[layer-1][i].getOutput();
			weight = weights[layer-1][i][neuronIndex];
			sum += output * weight;
		}
		return sum;
	}
	
	
	/**
	 * Returns the sum of the errors of the subsequent neurons, weighted
	 * Used for propagating errors back
	 * @param layer
	 * @param neuronIndex
	 * @return
	 */
	private double getTotalOutputError(int layer, int neuronIndex) {
		double sum = 0;
		for (int to = 0; to < weights[layer][neuronIndex].length; to++) {
			sum += weights[layer][neuronIndex][to]
					* neurons[layer + 1][to].getError();
	}
		return sum;
	}
	
	/**
	 * Prints the weights from each of this Neural network's layers
	 */
	void printWeights() {
		log.info("Weights:");
		StringBuilder sb = new StringBuilder();
		sb.append("\r\n");		
		for (int layer = 0; layer < weights.length; layer++) {
			log.info("Layer " + layer + "-" + (layer + 1));
			for (int from = 0; from < weights[layer].length; from++) {
				for (int to = 0; to < weights[layer][from].length; to++) {
					sb.append(weights[layer][from][to]);
					sb.append(" ");
					
				}
				sb.append("\r\n");
			}
			log.info(sb.toString());
			sb.delete(0,  sb.length() - 1);			
			log.info("");
		}
	}
	
	
	/**
	 * Prints the state of the neurons
	 */
	void printNeurons() {
		for (int layer = 0; layer < neurons.length; layer++) {
			log.info("Layer " + layer);
			for (int i = 0; i < neurons[layer].length; i++) {
				log.info("Neuron " + i + ": " + neurons[layer][i].getInput()
						+ " / " + neurons[layer][i].getError() + " / "
						+ neurons[layer][i].getOutput());
			}
			log.info("");
		}
	}
	
	/**
	 * Computes the error in the output layer neurons
	 * @param input The input value for this output-layer neuron
	 * @param prediction The prediction of the neural network
	 * @param truth The actual value of this training example
	 * @return Returns the error value
	 */
	private double computeOutputError(double input, double prediction,
			double truth) {
		return errorFunction.getErrorDerivative(input) * (truth - prediction);
	}	
	
	/**
	 * Returns how many examples have been loaded for the given mode
	 * @param mode The mode to check for
	 * @return The number of examples that have been loaded for this mode
	 */
	int getNumExamples(Mode mode) {
		switch (mode) {
			case TRAINING: {
				return trainingExamples.size();
			}
			case TESTING: {
				return testingExamples.size();
			}
		}
		return -1;
	}
}
