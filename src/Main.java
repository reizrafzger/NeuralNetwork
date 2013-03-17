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
			nn.loadExamples("var/abaloneNormalized.csv");
		} catch (IOException e) {
			log.error(e);
		}
		
		for (int i = 0; i < 1000; i++) {
			nn.trainAllExamples(0.25);
		}
		nn.printWeights();
		nn.printNeurons();

	}

}
