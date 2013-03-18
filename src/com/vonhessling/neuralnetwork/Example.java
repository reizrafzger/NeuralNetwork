package com.vonhessling.neuralnetwork;

/**
 * One data point, used e.g. for training or testing
 * @author vonhessling
 *
 */
public class Example {
	
	// assumption: input and output is normalized between 0 and 1
	private double input[];
	private double output[];

	public Example(double[] input, double[] output) {
		this.input = input;
		this.output = output;		
	}

	
	public double[] getInput() {
		return input;
	}

	public double[] getOutput() {
		return output;
	}
	
	public void setOutput(double[] output) {
		this.output = output;
	}

	public void setInput(double[] input) {
		this.input = input;
	}
	
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("Example {");
		for (int i = 0; i < input.length; i++) {
			sb.append(input[i]);
			sb.append(" ");
		}
		sb.append("} - {");
		for (int i = 0; i < output.length; i++) {
			sb.append(output[i]);
		}
		sb.append("}");
		return sb.toString();
	}

}
