package com.vonhessling.neuralnetwork;
import java.io.File;
import java.io.IOException;
import java.util.*;
import org.apache.log4j.Logger;
import org.apache.commons.io.FileUtils;

/**
 * Implementation of a feed-forward backpropagation neural network
 * 
 * @TODO needs momentum and prediction functionality framework
 * 
 * @author vonhessling
 *
 */
public class NeuralNetwork {
    private static final Logger log = Logger.getLogger(NeuralNetwork.class);
	
	// [layerIndex][fromNeuron][toNeuron]
	private double[][][] weights; 

	// [layerIndex][neuronIndex]
	private Neuron[][] neurons;
	
	// training examples
	private List<Example> examples;
	
	/**
	 * 
	 * @param learningRate The learning rate
	 * @param numNeurons Describes the number of neurons per layer; Index is the layer index
	 */
	public NeuralNetwork(int[] numNeurons) {
		this.examples = new ArrayList<Example>();
		
		// initialize neurons in all layers
		neurons = new Neuron[numNeurons.length][];
		for (int layer = 0; layer < numNeurons.length; layer++) {
			neurons[layer] = new Neuron[numNeurons[layer]];
			for (int neuronIndex = 0; neuronIndex < neurons[layer].length; neuronIndex++) {
				neurons[layer][neuronIndex] = new Neuron();
			}
		}
		
		// initialize weights
		weights = new double[numNeurons.length - 1][][]; // there is one less weight than layers
		for (int layer = 0; layer < numNeurons.length - 1; layer++) {
			weights[layer] = new double[numNeurons[layer] + 1]    // "from" (+ 1 to add a bias weight to every neuron's input)
										[numNeurons[layer + 1]];  // "to"
			for (int from = 0; from < weights[layer].length; from++) {
				for (int to = 0; to < weights[layer][from].length; to++) {
					if (from == weights[layer].length - 1) {
						weights[layer][from][to] = -1; // bias weight in the last from column
					} else {
						weights[layer][from][to] = Math.random(); // initialize regular weights randomly
					}
					
				}
			}
		}
	}

	/**
	 * Loads the examples in the given file
	 * @param fileName The name of the file
	 * @throws IOException Throws IOException if i/o error occurs with file
	 */
	void loadExamples(String fileName) throws IOException {
		log.info("Loading examples...");
		
		List<String> lines = FileUtils.readLines(new File(fileName));
		for (String line : lines) {
			double[] inputVector = new double[neurons[0].length];                   // number of neurons on layer 0
			double[] outputVector = new double[neurons[neurons.length - 1].length]; // number of output neurons
			String[] values = line.split(",");
			if (values.length != inputVector.length + outputVector.length) {
				throw new IOException("Line contains " + values.length + " items, not " + (inputVector.length + outputVector.length) + " as required. Line content: " + line);
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
			examples.add(new Example(inputVector, outputVector));
			
		}
		log.info(examples.size() + " examples loaded.");
	}
	
	/**
	 * Trains the neural network using all loaded examples once.
	 * @param learningRate The learning rate to use.
	 */
	public void trainAllExamples(double learningRate) {
		log.info("Training Neural Network...");
		for (Example example : examples) {
			// set input-neuron's output value to the example's input vector
			for (int j = 0; j < neurons[0].length; j++) { 
				neurons[0][j].setOutput(example.getInput()[j]);
			}
			
			Neuron currentNeuron;
			// propagate the value forward
			for (int layer = 1; layer < neurons.length; layer++) { // for each layer
				for (int i = 0; i < neurons[layer].length; i++) {  // update each neuron's input value
					currentNeuron = neurons[layer][i];
					// find input value by summing up the output of the previous layer, weighted
					currentNeuron.setInput(getInputStrengthForNeuron(layer, i));
					// this neuron's output is the sigmoid value of the input
					currentNeuron.setOutput(sigmoid(currentNeuron.getInput()));

					if (layer == neurons.length - 1) { // for the output layer only, compute the error
						currentNeuron.setError(
								computeOutputError(
										currentNeuron.getInput(), 
										currentNeuron.getOutput(), 
										example.getOutput()[i]));
					}
				}
			}
			


			// propagate error back layer by layer and update weights accordingly
			for (int layer = neurons.length - 2; layer >= 0; layer--) { // for each layer, backwards
				for (int j = 0; j < neurons[layer].length; j++) {
					currentNeuron = neurons[layer][j];
					currentNeuron.setError(	
							sigmoidDerivative(currentNeuron.getInput()) * 
								getTotalOutputError(layer, j)); // update error in current layer
					for (int to = 0; to < weights[layer][j].length; to++) {
						weights[layer][j][to] = weights[layer][j][to] + 
								learningRate * currentNeuron.getOutput() * neurons[layer+1][to].getError();
					}
				}
			}	
		}

	}
	
	
	
	/**
	 * Returns the input strength of the given neuron in the given layer by 
	 * looking up the previous layer's output and the weights
	 * Used for initialization 
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
			sum += weights[layer][neuronIndex][to] * neurons[layer+1][to].getError();
		}
		return sum;
	}
	
	
	
	/**
	 * Prints the weights from each of this Neural network's layers
	 */
	public void printWeights() {
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
	
	
	public void printNeurons() {
		for (int layer = 0; layer < neurons.length; layer++) {
			log.info("Layer " + layer);
			for (int i = 0; i < neurons[layer].length; i++) {
				log.info("Neuron " + i + ": " + neurons[layer][i].getInput() + " / " + neurons[layer][i].getError() + " / " + neurons[layer][i].getOutput());
			}
			log.info("");
		}
	}
	
	/**
	 * Sigmoid function
	 * @param x
	 * @return
	 */
	public static double sigmoid(double x) {
		return (1 / ( 1 + Math.pow(Math.E,(-1 * x))));
	}
	
	/**
	 * Derivate of sigmoid function.
	 * @param x
	 * @return
	 */
	public static double sigmoidDerivative(double x) {
		double s = sigmoid(x);
		return s * (1-s);		
	}	
	
	/**
	 * Computes the error in the output layer neurons
	 * @param input
	 * @param prediction
	 * @param truth
	 * @return
	 */
	private double computeOutputError(double input, double prediction, double truth) {
		return sigmoidDerivative(input) * (truth - prediction);
	}	
	
}
