
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

}
