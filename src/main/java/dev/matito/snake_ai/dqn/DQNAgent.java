package dev.matito.snake_ai.dqn;

import dev.matito.snake_ai.Direction;
import dev.matito.snake_ai.GameConfig;
import dev.matito.snake_ai.SnakeGame;

import java.io.*;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Random;

public final class DQNAgent {
	private static final int NUM_ACTIONS = 3;
	private static final int HIDDEN_SIZE_1 = 128;
	private static final int HIDDEN_SIZE_2 = 64;
	private static final String DEFAULT_WEIGHTS_FILE = "dqn_weights.bin";

	private final NeuralNetwork onlineNetwork;
	private final NeuralNetwork targetNetwork;
	private final ReplayBuffer replayBuffer;
	private final Random random;
	private final File persistenceFile;

	private double alpha = 0.001;
	private double gamma = 0.95;
	private double epsilon;
	private double epsilonMin;
	private double epsilonDecay;

	private int targetUpdateFrequency = 1000;
	private int updateCounter = 0;

	private int autoSaveEverySteps = 500;
	private long autoSaveEveryMillis = 5000L;
	private long steps = 0L;
	private long lastAutoSaveMillis = 0L;

	private DQNState prevState = null;
	private int prevAction = -1;
	private int prevScore = 0;

	public DQNAgent(int gridWidth, int gridHeight, int replayCapacity, long seed) {
		GameConfig config = new GameConfig("config.properties");
		this.epsilon = config.getAgentEpsilonStart();
		this.epsilonDecay = config.getEpsilonDecay();
		this.epsilonMin = config.getAgentEpsilonMin();
		int inputSize = 4 * gridWidth * gridHeight;
		this.onlineNetwork = new NeuralNetwork(inputSize, HIDDEN_SIZE_1, HIDDEN_SIZE_2, NUM_ACTIONS, seed);
		this.targetNetwork = new NeuralNetwork(inputSize, HIDDEN_SIZE_1, HIDDEN_SIZE_2, NUM_ACTIONS, seed + 1);
		this.targetNetwork.copyWeightsFrom(onlineNetwork);
		this.replayBuffer = new ReplayBuffer(replayCapacity, seed + 2);
		this.random = new Random(seed + 3);
		this.persistenceFile = new File(DEFAULT_WEIGHTS_FILE);

		loadIfExists();
		installShutdownHook();
		this.lastAutoSaveMillis = System.currentTimeMillis();
	}

	public void setAutoSave(int everySteps, long everyMillis) {
		this.autoSaveEverySteps = Math.max(0, everySteps);
		this.autoSaveEveryMillis = Math.max(0L, everyMillis);
	}

	public void resetEpisode() {
		prevState = null;
		prevAction = -1;
		prevScore = 0;
	}

	public Direction decide(SnakeGame game) {
		DQNState currentState = DQNState.fromGame(game);

		if (prevState != null && prevAction != -1) {
			double reward = computeReward(prevState, currentState);
			replayBuffer.add(new Transition(
					prevState.getGridData(),
					prevAction,
					reward,
					currentState.getGridData(),
					currentState.isTerminal()
			));
		}

		int action = selectAction(currentState);
		Direction outDir = applyRelativeAction(currentState.getDirection(), action);

		prevState = currentState;
		prevAction = action;
		prevScore = currentState.getScore();

		steps++;
		maybeAutoSave();

		return outDir;
	}

	public void train(int batchSize) {
		if (replayBuffer.size() < batchSize) {
			return;
		}

		List<Transition> batch = replayBuffer.sample(batchSize);

		for (Transition t : batch) {
			double[] qValues = onlineNetwork.forward(t.getState());
			double target;

			if (t.isTerminal()) {
				target = t.getReward();
			} else {
				double[] nextQValues = targetNetwork.forward(t.getNextState());
				double maxNextQ = nextQValues[0];
				for (int i = 1; i < NUM_ACTIONS; i++) {
					if (nextQValues[i] > maxNextQ) {
						maxNextQ = nextQValues[i];
					}
				}
				target = t.getReward() + gamma * maxNextQ;
			}

			double[] targetQValues = qValues.clone();
			targetQValues[t.getAction()] = target;

			backpropagate(t.getState(), targetQValues);
		}

		updateCounter++;
		if (updateCounter % targetUpdateFrequency == 0) {
			targetNetwork.copyWeightsFrom(onlineNetwork);
		}

		if (epsilon > epsilonMin) {
			epsilon = Math.max(epsilonMin, epsilon * epsilonDecay);
		}
	}

	private int selectAction(DQNState state) {
		if (random.nextDouble() < epsilon) {
			return random.nextInt(NUM_ACTIONS);
		}

		double[] qValues = onlineNetwork.forward(state.getGridData());
		int bestAction = 0;
		double bestValue = qValues[0];

		for (int i = 1; i < NUM_ACTIONS; i++) {
			if (qValues[i] > bestValue) {
				bestValue = qValues[i];
				bestAction = i;
			}
		}

		return bestAction;
	}

	private void backpropagate(double[] input, double[] targetOutput) {
		double[] output = onlineNetwork.forward(input);

		double[] outputError = new double[NUM_ACTIONS];
		for (int i = 0; i < NUM_ACTIONS; i++) {
			outputError[i] = targetOutput[i] - output[i];
		}

		double[] h2Error = new double[HIDDEN_SIZE_2];
		for (int i = 0; i < HIDDEN_SIZE_2; i++) {
			for (int j = 0; j < NUM_ACTIONS; j++) {
				h2Error[i] += outputError[j] * onlineNetwork.getW3()[j][i];
			}
			if (onlineNetwork.getH2()[i] <= 0) {
				h2Error[i] = 0;
			}
		}

		double[] h1Error = new double[HIDDEN_SIZE_1];
		for (int i = 0; i < HIDDEN_SIZE_1; i++) {
			for (int j = 0; j < HIDDEN_SIZE_2; j++) {
				h1Error[i] += h2Error[j] * onlineNetwork.getW2()[j][i];
			}
			if (onlineNetwork.getH1()[i] <= 0) {
				h1Error[i] = 0;
			}
		}

		double[][] w3 = onlineNetwork.getW3();
		double[] b3 = onlineNetwork.getB3();
		for (int i = 0; i < NUM_ACTIONS; i++) {
			for (int j = 0; j < HIDDEN_SIZE_2; j++) {
				w3[i][j] += alpha * outputError[i] * onlineNetwork.getH2()[j];
			}
			b3[i] += alpha * outputError[i];
		}

		double[][] w2 = onlineNetwork.getW2();
		double[] b2 = onlineNetwork.getB2();
		for (int i = 0; i < HIDDEN_SIZE_2; i++) {
			for (int j = 0; j < HIDDEN_SIZE_1; j++) {
				w2[i][j] += alpha * h2Error[i] * onlineNetwork.getH1()[j];
			}
			b2[i] += alpha * h2Error[i];
		}

		double[][] w1 = onlineNetwork.getW1();
		double[] b1 = onlineNetwork.getB1();
		for (int i = 0; i < HIDDEN_SIZE_1; i++) {
			for (int j = 0; j < input.length; j++) {
				w1[i][j] += alpha * h1Error[i] * input[j];
			}
			b1[i] += alpha * h1Error[i];
		}
	}

	private double computeReward(DQNState prev, DQNState current) {
		if (current.isTerminal()) {
			return -10.0;
		}

		if (current.getScore() > prev.getScore()) {
			return 10.0;
		}

		return -0.01;
	}

	private Direction applyRelativeAction(Direction current, int action) {
		if (action == 0) return current;
		if (action == 1) return turnLeft(current);
		return turnRight(current);
	}

	private Direction turnLeft(Direction d) {
		return switch (d) {
			case UP -> Direction.LEFT;
			case LEFT -> Direction.DOWN;
			case DOWN -> Direction.RIGHT;
			case RIGHT -> Direction.UP;
		};
	}

	private Direction turnRight(Direction d) {
		return switch (d) {
			case UP -> Direction.RIGHT;
			case RIGHT -> Direction.DOWN;
			case DOWN -> Direction.LEFT;
			case LEFT -> Direction.UP;
		};
	}

	private void maybeAutoSave() {
		if (persistenceFile == null) return;

		long now = System.currentTimeMillis();
		boolean bySteps = autoSaveEverySteps > 0 && (steps % autoSaveEverySteps) == 0;
		boolean byTime = autoSaveEveryMillis > 0 && (now - lastAutoSaveMillis) >= autoSaveEveryMillis;
		if (!bySteps && !byTime) return;

		try {
			saveAtomic(persistenceFile);
			lastAutoSaveMillis = now;
		} catch (IOException ignored) {
		}
	}

	private void loadIfExists() {
		if (persistenceFile == null) return;
		if (!persistenceFile.isFile()) return;
		try {
			load(persistenceFile);
		} catch (IOException ignored) {
		}
	}

	private void installShutdownHook() {
		if (persistenceFile == null) return;
		Runtime.getRuntime().addShutdownHook(new Thread(() -> {
			try {
				saveAtomic(persistenceFile);
			} catch (IOException ignored) {
			}
		}, "dqnSave"));
	}

	private void saveAtomic(File file) throws IOException {
		Path target = file.toPath();
		Path dir = target.getParent();
		if (dir != null) Files.createDirectories(dir);
		Path tmp = target.resolveSibling(target.getFileName().toString() + ".tmp");
		save(tmp.toFile());
		try {
			Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
		} catch (AtomicMoveNotSupportedException e) {
			Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING);
		}
	}

	public void save(File file) throws IOException {
		try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)))) {
			out.writeDouble(alpha);
			out.writeDouble(gamma);
			out.writeDouble(epsilon);
			out.writeDouble(epsilonMin);
			out.writeDouble(epsilonDecay);
			out.writeInt(targetUpdateFrequency);
			out.writeInt(updateCounter);

			onlineNetwork.save(out);
			targetNetwork.save(out);
		}
	}

	public void load(File file) throws IOException {
		try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
			alpha = in.readDouble();
			gamma = in.readDouble();
			epsilon = in.readDouble();
			epsilonMin = in.readDouble();
			epsilonDecay = in.readDouble();
			targetUpdateFrequency = in.readInt();
			updateCounter = in.readInt();

			onlineNetwork.load(in);
			targetNetwork.load(in);
			System.out.println("loaded from file");
		}
	}

	public void setLearningRate(double alpha) {
		this.alpha = Math.max(0.0, Math.min(1.0, alpha));
	}

	public void setGamma(double gamma) {
		this.gamma = Math.max(0.0, Math.min(1.0, gamma));
	}

	public void setEpsilon(double epsilon, double epsilonMin, double epsilonDecay) {
		this.epsilon = Math.max(0.0, Math.min(1.0, epsilon));
		this.epsilonMin = Math.max(0.0, Math.min(1.0, epsilonMin));
		this.epsilonDecay = Math.max(0.0, Math.min(1.0, epsilonDecay));
	}

	public void setTargetUpdateFrequency(int frequency) {
		this.targetUpdateFrequency = Math.max(1, frequency);
	}

	public double getEpsilon() {
		return epsilon;
	}

	public int getReplayBufferSize() {
		return replayBuffer.size();
	}
}
