package dev.matito.snake_ai.dqn;

import java.io.*;
import java.util.Random;

public final class NeuralNetwork {
	private final int inputSize;
	private final int hiddenSize1;
	private final int hiddenSize2;
	private final int outputSize;

	private double[][] w1;
	private double[] b1;
	private double[][] w2;
	private double[] b2;
	private double[][] w3;
	private double[] b3;

	private double[] h1;
	private double[] h2;

	public NeuralNetwork(int inputSize, int hiddenSize1, int hiddenSize2, int outputSize, long seed) {
		this.inputSize = inputSize;
		this.hiddenSize1 = hiddenSize1;
		this.hiddenSize2 = hiddenSize2;
		this.outputSize = outputSize;

		Random rnd = new Random(seed);
		double scale1 = Math.sqrt(2.0 / inputSize);
		double scale2 = Math.sqrt(2.0 / hiddenSize1);
		double scale3 = Math.sqrt(2.0 / hiddenSize2);

		w1 = new double[hiddenSize1][inputSize];
		b1 = new double[hiddenSize1];
		w2 = new double[hiddenSize2][hiddenSize1];
		b2 = new double[hiddenSize2];
		w3 = new double[outputSize][hiddenSize2];
		b3 = new double[outputSize];

		for (int i = 0; i < hiddenSize1; i++) {
			for (int j = 0; j < inputSize; j++) {
				w1[i][j] = rnd.nextGaussian() * scale1;
			}
		}

		for (int i = 0; i < hiddenSize2; i++) {
			for (int j = 0; j < hiddenSize1; j++) {
				w2[i][j] = rnd.nextGaussian() * scale2;
			}
		}

		for (int i = 0; i < outputSize; i++) {
			for (int j = 0; j < hiddenSize2; j++) {
				w3[i][j] = rnd.nextGaussian() * scale3;
			}
		}

		h1 = new double[hiddenSize1];
		h2 = new double[hiddenSize2];
	}

	public double[] forward(double[] input) {
		for (int i = 0; i < hiddenSize1; i++) {
			h1[i] = b1[i];
			for (int j = 0; j < inputSize; j++) {
				h1[i] += w1[i][j] * input[j];
			}
			h1[i] = relu(h1[i]);
		}

		for (int i = 0; i < hiddenSize2; i++) {
			h2[i] = b2[i];
			for (int j = 0; j < hiddenSize1; j++) {
				h2[i] += w2[i][j] * h1[j];
			}
			h2[i] = relu(h2[i]);
		}

		double[] output = new double[outputSize];
		for (int i = 0; i < outputSize; i++) {
			output[i] = b3[i];
			for (int j = 0; j < hiddenSize2; j++) {
				output[i] += w3[i][j] * h2[j];
			}
		}

		return output;
	}

	public void updateWeights(double[][] w1Delta, double[] b1Delta,
							  double[][] w2Delta, double[] b2Delta,
							  double[][] w3Delta, double[] b3Delta) {
		for (int i = 0; i < hiddenSize1; i++) {
			for (int j = 0; j < inputSize; j++) {
				w1[i][j] += w1Delta[i][j];
			}
			b1[i] += b1Delta[i];
		}

		for (int i = 0; i < hiddenSize2; i++) {
			for (int j = 0; j < hiddenSize1; j++) {
				w2[i][j] += w2Delta[i][j];
			}
			b2[i] += b2Delta[i];
		}

		for (int i = 0; i < outputSize; i++) {
			for (int j = 0; j < hiddenSize2; j++) {
				w3[i][j] += w3Delta[i][j];
			}
			b3[i] += b3Delta[i];
		}
	}

	public void copyWeightsFrom(NeuralNetwork other) {
		for (int i = 0; i < hiddenSize1; i++) {
			System.arraycopy(other.w1[i], 0, w1[i], 0, inputSize);
		}
		System.arraycopy(other.b1, 0, b1, 0, hiddenSize1);

		for (int i = 0; i < hiddenSize2; i++) {
			System.arraycopy(other.w2[i], 0, w2[i], 0, hiddenSize1);
		}
		System.arraycopy(other.b2, 0, b2, 0, hiddenSize2);

		for (int i = 0; i < outputSize; i++) {
			System.arraycopy(other.w3[i], 0, w3[i], 0, hiddenSize2);
		}
		System.arraycopy(other.b3, 0, b3, 0, outputSize);
	}

	public void save(DataOutputStream out) throws IOException {
		out.writeInt(inputSize);
		out.writeInt(hiddenSize1);
		out.writeInt(hiddenSize2);
		out.writeInt(outputSize);

		for (int i = 0; i < hiddenSize1; i++) {
			for (int j = 0; j < inputSize; j++) {
				out.writeDouble(w1[i][j]);
			}
		}
		for (int i = 0; i < hiddenSize1; i++) {
			out.writeDouble(b1[i]);
		}

		for (int i = 0; i < hiddenSize2; i++) {
			for (int j = 0; j < hiddenSize1; j++) {
				out.writeDouble(w2[i][j]);
			}
		}
		for (int i = 0; i < hiddenSize2; i++) {
			out.writeDouble(b2[i]);
		}

		for (int i = 0; i < outputSize; i++) {
			for (int j = 0; j < hiddenSize2; j++) {
				out.writeDouble(w3[i][j]);
			}
		}
		for (int i = 0; i < outputSize; i++) {
			out.writeDouble(b3[i]);
		}
	}

	public void load(DataInputStream in) throws IOException {
		int loadedInputSize = in.readInt();
		int loadedHiddenSize1 = in.readInt();
		int loadedHiddenSize2 = in.readInt();
		int loadedOutputSize = in.readInt();

		if (loadedInputSize != inputSize || loadedHiddenSize1 != hiddenSize1 ||
				loadedHiddenSize2 != hiddenSize2 || loadedOutputSize != outputSize) {
			throw new IOException("Network dimension mismatch");
		}

		for (int i = 0; i < hiddenSize1; i++) {
			for (int j = 0; j < inputSize; j++) {
				w1[i][j] = in.readDouble();
			}
		}
		for (int i = 0; i < hiddenSize1; i++) {
			b1[i] = in.readDouble();
		}

		for (int i = 0; i < hiddenSize2; i++) {
			for (int j = 0; j < hiddenSize1; j++) {
				w2[i][j] = in.readDouble();
			}
		}
		for (int i = 0; i < hiddenSize2; i++) {
			b2[i] = in.readDouble();
		}

		for (int i = 0; i < outputSize; i++) {
			for (int j = 0; j < hiddenSize2; j++) {
				w3[i][j] = in.readDouble();
			}
		}
		for (int i = 0; i < outputSize; i++) {
			b3[i] = in.readDouble();
		}
	}

	private double relu(double x) {
		return x > 0 ? x : 0;
	}

	double[][] getW1() { return w1; }
	double[] getB1() { return b1; }
	double[][] getW2() { return w2; }
	double[] getB2() { return b2; }
	double[][] getW3() { return w3; }
	double[] getB3() { return b3; }
	double[] getH1() { return h1; }
	double[] getH2() { return h2; }
}