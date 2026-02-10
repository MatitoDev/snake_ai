package dev.matito.snake_ai.dqn;

import dev.matito.snake_ai.Direction;
import dev.matito.snake_ai.Config;
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

	private int targetUpdateFrequency = 100;
	private int updateCounter = 0;

	private int autoSaveEverySteps = 500;
	private long autoSaveEveryMillis = 5000L;
	private long steps = 0L;
	private long lastAutoSaveMillis = 0L;

	private DQNState prevState = null;
	private int prevAction = -1;
	private int prevScore = 0;

	private volatile double[] lastQValues = null;
	private volatile long lastSummaryMillis = 0L;
	private volatile String lastSummaryJson = null;
	private final int inputSize;

	public DQNAgent(int gridWidth, int gridHeight, int replayCapacity, long seed) {
		Config config = new Config("config.properties");
		this.inputSize = 4 * gridWidth * gridHeight;
		this.onlineNetwork = new NeuralNetwork(inputSize, HIDDEN_SIZE_1, HIDDEN_SIZE_2, NUM_ACTIONS, seed);
		this.targetNetwork = new NeuralNetwork(inputSize, HIDDEN_SIZE_1, HIDDEN_SIZE_2, NUM_ACTIONS, seed + 1);

		this.replayBuffer = new ReplayBuffer(replayCapacity, seed + 2);
		this.persistenceFile = new File(DEFAULT_WEIGHTS_FILE);
		boolean loaded = loadIfExists();
		if (config.isEpsilonOverride() || !loaded) {
			this.epsilon = config.getAgentEpsilonStart();
			this.epsilonDecay = config.getEpsilonDecay();
			this.epsilonMin = config.getAgentEpsilonMin();
		}
		this.targetNetwork.copyWeightsFrom(onlineNetwork);
		this.random = new Random(seed + 3);
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
		lastQValues = qValues.clone();
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
		if (current.isTooMuchSteps()) return -200;
		if (current.isTerminal()) return -40.0;
		if (current.getScore() > prev.getScore()) return 50.0;

		if (prev.getFoodDistance() == Integer.MIN_VALUE) {
			return -0.01;
		}
		if (current.getFoodDistance() < prev.getFoodDistance()) {
			return +0.2;
		}
		if (current.getFoodDistance() > prev.getFoodDistance()) {
			return -0.2;
		}

		return -0.1;
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

	private boolean loadIfExists() {
		if (persistenceFile == null) {
			if (new Config("config.properties").isEpsilonOverride()) System.out.println("WARNING: epsilon override is enabled but no weights file specified!");
			return false;
		};
		if (!persistenceFile.isFile()) return false;
		try {
			load(persistenceFile);
		} catch (IOException ignored) {
			return false;
		}
		return true;
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

	public double getEpsilonMin() {
		return epsilonMin;
	}

	public int getReplayBufferSize() {
		return replayBuffer.size();
	}

	public double getLearningRate() { return alpha; }
	public double getGamma() { return gamma; }
	public double[] getLastQValues() { return lastQValues == null ? null : lastQValues.clone(); }

	public String getOnlineNetworkSummaryJson() {
		long now = System.currentTimeMillis();
		String cached = lastSummaryJson;
		if (cached != null && (now - lastSummaryMillis) < 1500L) return cached;
		String j = buildNetworkSummaryJson(onlineNetwork);
		lastSummaryJson = j;
		lastSummaryMillis = now;
		return j;
	}

	private static String buildNetworkSummaryJson(NeuralNetwork n) {
		StringBuilder sb = new StringBuilder(1024);
		sb.append('{');
		sb.append("\"w1\":"); appendMatrixStats(sb, n.getW1()); sb.append(',');
		sb.append("\"b1\":"); appendArrayStats(sb, n.getB1()); sb.append(',');
		sb.append("\"w2\":"); appendMatrixStats(sb, n.getW2()); sb.append(',');
		sb.append("\"b2\":"); appendArrayStats(sb, n.getB2()); sb.append(',');
		sb.append("\"w3\":"); appendMatrixStats(sb, n.getW3()); sb.append(',');
		sb.append("\"b3\":"); appendArrayStats(sb, n.getB3());
		sb.append('}');
		return sb.toString();
	}

	private static void appendMatrixStats(StringBuilder sb, double[][] m) {
		Stats s = new Stats();
		for (double[] row : m) for (double v : row) s.add(v);
		appendStatsObj(sb, s);
	}

	private static void appendArrayStats(StringBuilder sb, double[] a) {
		Stats s = new Stats();
		for (double v : a) s.add(v);
		appendStatsObj(sb, s);
	}

	private static void appendStatsObj(StringBuilder sb, Stats s) {
		sb.append('{');
		sb.append("\"n\":").append(s.n).append(',');
		sb.append("\"min\":").append(s.min).append(',');
		sb.append("\"max\":").append(s.max).append(',');
		sb.append("\"mean\":").append(s.mean).append(',');
		sb.append("\"std\":").append(s.std());
		sb.append('}');
	}

	private static final class Stats {
		long n = 0L;
		double mean = 0.0;
		double m2 = 0.0;
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		void add(double x) {
			n++;
			if (x < min) min = x;
			if (x > max) max = x;
			double delta = x - mean;
			mean += delta / n;
			double delta2 = x - mean;
			m2 += delta * delta2;
		}
		double std() {
			if (n <= 1) return 0.0;
			double var = m2 / (n - 1);
			return Math.sqrt(Math.max(0.0, var));
		}
	}

	public String getNetworkSliceJson(
			String net,
			int inStart, int inCount,
			int h1Start, int h1Count,
			int h2Start, int h2Count
	) {
		NeuralNetwork n = "target".equalsIgnoreCase(net) ? targetNetwork : onlineNetwork;

		inCount = clampInt(inCount, 1, 32);
		h1Count = clampInt(h1Count, 1, 32);
		h2Count = clampInt(h2Count, 1, 32);

		inStart = clampInt(inStart, 0, inputSize - 1);
		h1Start = clampInt(h1Start, 0, HIDDEN_SIZE_1 - 1);
		h2Start = clampInt(h2Start, 0, HIDDEN_SIZE_2 - 1);

		int inEnd = Math.min(inputSize, inStart + inCount);
		int h1End = Math.min(HIDDEN_SIZE_1, h1Start + h1Count);
		int h2End = Math.min(HIDDEN_SIZE_2, h2Start + h2Count);

		double[][] w1 = n.getW1();
		double[] b1 = n.getB1();
		double[][] w2 = n.getW2();
		double[] b2 = n.getB2();
		double[][] w3 = n.getW3();
		double[] b3 = n.getB3();

		StringBuilder sb = new StringBuilder(80_000);
		sb.append('{');

		sb.append("\"net\":\"").append(jsonEscape(net == null ? "online" : net)).append("\",");

		sb.append("\"sizes\":{");
		sb.append("\"input\":").append(inputSize).append(',');
		sb.append("\"h1\":").append(HIDDEN_SIZE_1).append(',');
		sb.append("\"h2\":").append(HIDDEN_SIZE_2).append(',');
		sb.append("\"out\":").append(NUM_ACTIONS);
		sb.append("},");

		sb.append("\"slice\":{");
		sb.append("\"inStart\":").append(inStart).append(',');
		sb.append("\"inCount\":").append(inEnd - inStart).append(',');
		sb.append("\"h1Start\":").append(h1Start).append(',');
		sb.append("\"h1Count\":").append(h1End - h1Start).append(',');
		sb.append("\"h2Start\":").append(h2Start).append(',');
		sb.append("\"h2Count\":").append(h2End - h2Start);
		sb.append("},");

		// b1
		sb.append("\"b1\":[");
		for (int i = h1Start; i < h1End; i++) {
			if (i > h1Start) sb.append(',');
			sb.append(b1[i]);
		}
		sb.append("],");

		// w1: [h1Count][inCount]
		sb.append("\"w1\":[");
		for (int i = h1Start; i < h1End; i++) {
			if (i > h1Start) sb.append(',');
			sb.append('[');
			for (int j = inStart; j < inEnd; j++) {
				if (j > inStart) sb.append(',');
				sb.append(w1[i][j]);
			}
			sb.append(']');
		}
		sb.append("],");

		// b2
		sb.append("\"b2\":[");
		for (int i = h2Start; i < h2End; i++) {
			if (i > h2Start) sb.append(',');
			sb.append(b2[i]);
		}
		sb.append("],");

		// w2: [h2Count][h1Count]
		sb.append("\"w2\":[");
		for (int i = h2Start; i < h2End; i++) {
			if (i > h2Start) sb.append(',');
			sb.append('[');
			for (int j = h1Start; j < h1End; j++) {
				if (j > h1Start) sb.append(',');
				sb.append(w2[i][j]);
			}
			sb.append(']');
		}
		sb.append("],");

		// b3
		sb.append("\"b3\":[");
		for (int i = 0; i < NUM_ACTIONS; i++) {
			if (i > 0) sb.append(',');
			sb.append(b3[i]);
		}
		sb.append("],");

		// w3: [out][h2Count]
		sb.append("\"w3\":[");
		for (int i = 0; i < NUM_ACTIONS; i++) {
			if (i > 0) sb.append(',');
			sb.append('[');
			for (int j = h2Start; j < h2End; j++) {
				if (j > h2Start) sb.append(',');
				sb.append(w3[i][j]);
			}
			sb.append(']');
		}
		sb.append("]");

		sb.append('}');
		return sb.toString();
	}

	private static int clampInt(int v, int lo, int hi) {
		if (v < lo) return lo;
		if (v > hi) return hi;
		return v;
	}

	private static String jsonEscape(String s) {
		if (s == null) return "";
		StringBuilder sb = new StringBuilder(s.length() + 16);
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			switch (c) {
				case '"' -> sb.append("\\\"");
				case '\\' -> sb.append("\\\\");
				case '\n' -> sb.append("\\n");
				case '\r' -> sb.append("\\r");
				case '\t' -> sb.append("\\t");
				default -> {
					if (c < 0x20) sb.append(' ');
					else sb.append(c);
				}
			}
		}
		return sb.toString();
	}


}
