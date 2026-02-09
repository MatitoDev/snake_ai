package dev.matito.snake_ai.ql;

import dev.matito.snake_ai.Direction;
import dev.matito.snake_ai.GameConfig;
import dev.matito.snake_ai.SnakeGame;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Random;

/**
 * Tabular Q learning agent.
 *
 * Actions are relative to the current direction:
 * 0 = straight, 1 = left, 2 = right.
 */
public final class QLAgent {
	private static final int NUM_STATES = 1 << 11; // 2048
	private static final int NUM_ACTIONS = 3;
	private static final String DEFAULT_QTABLE_FILE = "qtable.bin";

	private final double[][] q = new double[NUM_STATES][NUM_ACTIONS];
	private final Random rnd;
	private final File persistenceFile;

	private int autoSaveEverySteps = 200;
	private long autoSaveEveryMillis = 2000L;
	private long steps = 0L;
	private long lastAutoSaveMillis = 0L;

	private double alpha = 0.10;
	private double gamma = 0.90;

	private double epsilon;
	private double epsilonMin;
	private double epsilonDecay;

	private int prevStateId = -1;
	private int prevAction = -1;
	private int prevDistToFood = Integer.MIN_VALUE;
	private int prevScore = Integer.MIN_VALUE;
	private boolean prevTerminal;

	public QLAgent() {
		this(123456789L);
	}

	public QLAgent(long seed) {
		GameConfig config = new GameConfig("config.properties");
		this.epsilon = config.getAgentEpsilonStart();
		this.epsilonDecay = config.getEpsilonDecay();
		this.epsilonMin = config.getAgentEpsilonMin();
		this.rnd = new Random(seed);
		this.persistenceFile = new File(DEFAULT_QTABLE_FILE);
		loadIfExists();
		installShutdownHook();
		this.lastAutoSaveMillis = System.currentTimeMillis();
	}

	public void setAutoSave(int everySteps, long everyMillis) {
		this.autoSaveEverySteps = Math.max(0, everySteps);
		this.autoSaveEveryMillis = Math.max(0L, everyMillis);
	}

	/** Resets episode scoped memory (previous transition). */
	public void resetEpisode() {
		prevStateId = -1;
		prevAction = -1;
		prevDistToFood = Integer.MIN_VALUE;
		prevScore = Integer.MIN_VALUE;
		prevTerminal = false;
	}

	public void setLearning(double alpha, double gamma) {
		this.alpha = clamp(alpha, 0.0, 1.0);
		this.gamma = clamp(gamma, 0.0, 1.0);
	}

	public void setExploration(double epsilon, double epsilonMin, double epsilonDecay) {
		this.epsilon = clamp(epsilon, 0.0, 1.0);
		this.epsilonMin = clamp(epsilonMin, 0.0, 1.0);
		this.epsilonDecay = clamp(epsilonDecay, 0.0, 1.0);
	}

	/** Convenience: builds a {@link State} from the game and returns the next direction. */
	public Direction decide(SnakeGame game) {
		return decide(State.fromGame(game));
	}

	/** Main decision function. Updates Q for the previous transition, then selects the next action. */
	public Direction decide(State state) {
		int stateId = encode(state);

		// Q update for the previous step (s,a) -> s'
		if (prevStateId != -1 && prevAction != -1 && !prevTerminal) {
			double reward = rewardFromTransition(state);
			double target = state.isTerminal()
					? reward
					: reward + gamma * maxQ(stateId);
			q[prevStateId][prevAction] += alpha * (target - q[prevStateId][prevAction]);
		}

		int action = selectAction(stateId);
		Direction outDir = applyRelativeAction(state.getDirection(), action);

		prevStateId = stateId;
		prevAction = action;
		prevTerminal = state.isTerminal();
		prevScore = state.getScore();
		prevDistToFood = manhattan(state.getHeadX(), state.getHeadY(), state.getFood().getX(), state.getFood().getY());

		if (epsilon > epsilonMin) {
			epsilon = Math.max(epsilonMin, epsilon * epsilonDecay);
		}

		maybeAutoSave();

		return outDir;
	}

	// Backwards compatible aliases.
	public Direction getAI(State state) {
		return decide(state);
	}

	public Direction getAIDecision(State state) {
		return decide(state);
	}

	private int selectAction(int stateId) {
		if (rnd.nextDouble() < epsilon) {
			return rnd.nextInt(NUM_ACTIONS);
		}
		return argMaxAction(stateId);
	}

	private int argMaxAction(int stateId) {
		double best = Double.NEGATIVE_INFINITY;
		int bestA = 0;
		int ties = 0;

		for (int a = 0; a < NUM_ACTIONS; a++) {
			double v = q[stateId][a];
			if (v > best) {
				best = v;
				bestA = a;
				ties = 1;
			} else if (v == best) {
				ties++;
				if (rnd.nextInt(ties) == 0) {
					bestA = a;
				}
			}
		}
		return bestA;
	}

	private double maxQ(int stateId) {
		double best = q[stateId][0];
		for (int a = 1; a < NUM_ACTIONS; a++) {
			best = Math.max(best, q[stateId][a]);
		}
		return best;
	}

	private double rewardFromTransition(State s) {
		if (s.isTerminal()) {
			return -10.0;
		}

		boolean ateFood = (prevScore != Integer.MIN_VALUE) && (s.getScore() > prevScore);
		if (ateFood) {
			return +10.0;
		}

		int dist = manhattan(s.getHeadX(), s.getHeadY(), s.getFood().getX(), s.getFood().getY());
		if (prevDistToFood == Integer.MIN_VALUE) {
			return -0.01;
		}
		if (dist < prevDistToFood) {
			return +0.2;
		}
		if (dist > prevDistToFood) {
			return -0.2;
		}
		return -0.01;
	}

	/*
	 * 11 bits:
	 * 0 danger straight
	 * 1 danger left
	 * 2 danger right
	 * 3 food up
	 * 4 food right
	 * 5 food down
	 * 6 food left
	 * 7 dir up
	 * 8 dir right
	 * 9 dir down
	 * 10 dir left
	 */
	private int encode(State s) {
		Direction d = s.getDirection();

		boolean dangerStraight = isDanger(s, d);
		boolean dangerLeft = isDanger(s, turnLeft(d));
		boolean dangerRight = isDanger(s, turnRight(d));

		boolean foodUp = s.getFood().getY() < s.getHeadY();
		boolean foodDown = s.getFood().getY() > s.getHeadY();
		boolean foodLeft = s.getFood().getX() < s.getHeadX();
		boolean foodRight = s.getFood().getX() > s.getHeadX();

		int bits = 0;
		bits = putBit(bits, 0, dangerStraight);
		bits = putBit(bits, 1, dangerLeft);
		bits = putBit(bits, 2, dangerRight);

		bits = putBit(bits, 3, foodUp);
		bits = putBit(bits, 4, foodRight);
		bits = putBit(bits, 5, foodDown);
		bits = putBit(bits, 6, foodLeft);

		bits = putBit(bits, 7, d == Direction.UP);
		bits = putBit(bits, 8, d == Direction.RIGHT);
		bits = putBit(bits, 9, d == Direction.DOWN);
		bits = putBit(bits, 10, d == Direction.LEFT);

		return bits;
	}

	private boolean isDanger(State s, Direction moveDir) {
		int nx = s.getHeadX() + moveDir.getDx();
		int ny = s.getHeadY() + moveDir.getDy();

		if (!s.inBounds(nx, ny)) {
			return true;
		}

		// Moving into the current tail is safe because the tail moves away in the same tick.
		if (nx == s.getTailX() && ny == s.getTailY()) {
			return false;
		}

		return s.isOccupied(nx, ny);
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

	private int putBit(int x, int bit, boolean on) {
		return on ? (x | (1 << bit)) : x;
	}

	private int manhattan(int x1, int y1, int x2, int y2) {
		int dx = x1 - x2;
		if (dx < 0) dx = -dx;
		int dy = y1 - y2;
		if (dy < 0) dy = -dy;
		return dx + dy;
	}

	private double clamp(double v, double lo, double hi) {
		if (v < lo) return lo;
		if (v > hi) return hi;
		return v;
	}

	private void maybeAutoSave() {
		if (persistenceFile == null) return;

		steps++;
		long now = System.currentTimeMillis();
		boolean bySteps = autoSaveEverySteps > 0 && (steps % autoSaveEverySteps) == 0;
		boolean byTime = autoSaveEveryMillis > 0 && (now - lastAutoSaveMillis) >= autoSaveEveryMillis;
		if (!bySteps && !byTime) return;

		try {
			saveAtomic(persistenceFile);
			lastAutoSaveMillis = now;
		} catch (IOException ignored) {
			// ignore
		}
	}

	private void loadIfExists() {
		if (persistenceFile == null) return;
		if (!persistenceFile.isFile()) return;
		try {
			load(persistenceFile);
		} catch (IOException ignored) {
			// ignore
		}
	}

	private void installShutdownHook() {
		if (persistenceFile == null) return;
		Runtime.getRuntime().addShutdownHook(new Thread(() -> {
			try {
				saveAtomic(persistenceFile);
			} catch (IOException ignored) {
				// ignore
			}
		}, "qtableSave"));
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
			out.writeInt(NUM_STATES);
			out.writeInt(NUM_ACTIONS);
			out.writeDouble(alpha);
			out.writeDouble(gamma);
			out.writeDouble(epsilon);
			out.writeDouble(epsilonMin);
			out.writeDouble(epsilonDecay);

			for (int s = 0; s < NUM_STATES; s++) {
				for (int a = 0; a < NUM_ACTIONS; a++) {
					out.writeDouble(q[s][a]);
				}
			}
		}
	}

	public void load(File file) throws IOException {
		try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)))) {
			int states = in.readInt();
			int actions = in.readInt();
			if (states != NUM_STATES || actions != NUM_ACTIONS) {
				throw new IOException("Q table dimension mismatch: " + states + " x " + actions);
			}

			alpha = in.readDouble();
			gamma = in.readDouble();
			double epsilon_1 = in.readDouble();
			epsilonMin = in.readDouble();
			epsilonDecay = in.readDouble();

			for (int s = 0; s < NUM_STATES; s++) {
				for (int a = 0; a < NUM_ACTIONS; a++) {
					q[s][a] = in.readDouble();
				}
			}
		}
	}

	public double getEpsilon() {
		return epsilon;
	}
}
