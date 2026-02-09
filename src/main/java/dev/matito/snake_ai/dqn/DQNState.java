package dev.matito.snake_ai.dqn;

import dev.matito.snake_ai.Direction;
import dev.matito.snake_ai.GameState;
import dev.matito.snake_ai.Position;
import dev.matito.snake_ai.SnakeGame;

import java.util.List;

public final class DQNState {
	private final int width;
	private final int height;
	private final double[] gridData;
	private final Direction direction;
	private final boolean terminal;
	private final int score;

	private DQNState(int width, int height, double[] gridData, Direction direction, boolean terminal, int score) {
		this.width = width;
		this.height = height;
		this.gridData = gridData;
		this.direction = direction;
		this.terminal = terminal;
		this.score = score;
	}

	/*
		Vector hat weiter exakt 4 * w * h Werte (passt zu deinem aktuellen DQNAgent).
		Channel 0 nutzt die ersten 11 Slots als kompakte Features:
		  0 dangerStraight
		  1 dangerLeft
		  2 dangerRight
		  3 dirUp
		  4 dirRight
		  5 dirDown
		  6 dirLeft
		  7 foodUp
		  8 foodRight
		  9 foodDown
		  10 foodLeft

		Channel 1: body (ohne head)
		Channel 2: head
		Channel 3: food
	*/
	public static DQNState fromGame(SnakeGame game) {
		int w = game.getGridWidth();
		int h = game.getGridHeight();
		int planeSize = w * h;

		List<Position> snake = game.getSnakePositions();
		Position food = game.getFoodPosition();

		boolean terminal = game.getGameState() != GameState.RUNNING;
		int score = game.getScore();

		double[] v = new double[4 * planeSize];
		if (snake == null || snake.isEmpty()) {
			return new DQNState(w, h, v, Direction.RIGHT, terminal, score);
		}

		Position head = snake.getFirst();
		Direction dir = inferDirection(snake, w, h);

		boolean[] occNoTail = new boolean[planeSize];
		for (int i = 0; i < Math.max(0, snake.size() - 1); i++) {
			Position p = snake.get(i);
			int idx = p.getY() * w + p.getX();
			if (idx >= 0 && idx < planeSize) occNoTail[idx] = true;
		}

		// Body Channel (mit Tail, ohne Head)
		int bodyBase = planeSize;
		for (int i = 1; i < snake.size(); i++) {
			Position p = snake.get(i);
			int idx = p.getY() * w + p.getX();
			if (idx >= 0 && idx < planeSize) v[bodyBase + idx] = 1.0;
		}

		// Head Channel
		int headIdx = head.getY() * w + head.getX();
		if (headIdx >= 0 && headIdx < planeSize) v[2 * planeSize + headIdx] = 1.0;

		// Food Channel
		if (food != null) {
			int foodIdx = food.getY() * w + food.getX();
			if (foodIdx >= 0 && foodIdx < planeSize) v[3 * planeSize + foodIdx] = 1.0;
		}

		// Compact Features in Channel 0
		double dangerStraight = isDanger(head, dir, w, h, occNoTail) ? 1.0 : 0.0;
		double dangerLeft = isDanger(head, turnLeft(dir), w, h, occNoTail) ? 1.0 : 0.0;
		double dangerRight = isDanger(head, turnRight(dir), w, h, occNoTail) ? 1.0 : 0.0;

		put(v, 0, 0, dangerStraight);
		put(v, 0, 1, dangerLeft);
		put(v, 0, 2, dangerRight);

		put(v, 0, 3, dir == Direction.UP ? 1.0 : 0.0);
		put(v, 0, 4, dir == Direction.RIGHT ? 1.0 : 0.0);
		put(v, 0, 5, dir == Direction.DOWN ? 1.0 : 0.0);
		put(v, 0, 6, dir == Direction.LEFT ? 1.0 : 0.0);

		if (food != null) {
			put(v, 0, 7, food.getY() < head.getY() ? 1.0 : 0.0);   // foodUp
			put(v, 0, 8, food.getX() > head.getX() ? 1.0 : 0.0);   // foodRight
			put(v, 0, 9, food.getY() > head.getY() ? 1.0 : 0.0);   // foodDown
			put(v, 0, 10, food.getX() < head.getX() ? 1.0 : 0.0);  // foodLeft
		}

		return new DQNState(w, h, v, dir, terminal, score);
	}

	private static void put(double[] v, int base, int offset, double value) {
		int idx = base + offset;
		if (idx >= 0 && idx < v.length) v[idx] = value;
	}

	private static boolean isDanger(Position head, Direction moveDir, int w, int h, boolean[] occNoTail) {
		int nx = head.getX() + dx(moveDir);
		int ny = head.getY() + dy(moveDir);

		// Wand: out of bounds
		if (nx < 0 || nx >= w || ny < 0 || ny >= h) return true;

		int idx = ny * w + nx;
		if (idx < 0 || idx >= occNoTail.length) return true;

		// Koerper (ohne Tail)
		return occNoTail[idx];
	}

	private static int dx(Direction d) {
		return switch (d) {
			case LEFT -> -1;
			case RIGHT -> 1;
			default -> 0;
		};
	}

	private static int dy(Direction d) {
		return switch (d) {
			case UP -> -1;
			case DOWN -> 1;
			default -> 0;
		};
	}

	private static Direction turnLeft(Direction d) {
		return switch (d) {
			case UP -> Direction.LEFT;
			case LEFT -> Direction.DOWN;
			case DOWN -> Direction.RIGHT;
			case RIGHT -> Direction.UP;
		};
	}

	private static Direction turnRight(Direction d) {
		return switch (d) {
			case UP -> Direction.RIGHT;
			case RIGHT -> Direction.DOWN;
			case DOWN -> Direction.LEFT;
			case LEFT -> Direction.UP;
		};
	}

	private static Direction inferDirection(List<Position> snakeHeadFirst, int w, int h) {
		if (snakeHeadFirst == null || snakeHeadFirst.size() < 2) return Direction.RIGHT;

		Position head = snakeHeadFirst.get(0);
		Position neck = snakeHeadFirst.get(1);

		int dx = head.getX() - neck.getX();
		int dy = head.getY() - neck.getY();

		if (dx == 1 || dx == -(w - 1)) return Direction.RIGHT;
		if (dx == -1 || dx == (w - 1)) return Direction.LEFT;
		if (dy == 1 || dy == -(h - 1)) return Direction.DOWN;
		if (dy == -1 || dy == (h - 1)) return Direction.UP;

		// fallback
		return Direction.RIGHT;
	}

	public double[] getGridData() {
		return gridData;
	}

	public Direction getDirection() {
		return direction;
	}

	public boolean isTerminal() {
		return terminal;
	}

	public int getScore() {
		return score;
	}
}
