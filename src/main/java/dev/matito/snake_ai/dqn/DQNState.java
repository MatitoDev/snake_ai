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

	public static DQNState fromGame(SnakeGame game) {
		int w = game.getGridWidth();
		int h = game.getGridHeight();
		List<Position> snake = game.getSnakePositions();
		Position food = game.getFoodPosition();
		boolean terminal = game.getGameState() != GameState.RUNNING;
		int score = game.getScore();

		Direction dir = inferDirection(snake);

		double[] grid = new double[4 * w * h];

		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int idx = y * w + x;
				grid[idx] = (x == 0 || x == w - 1 || y == 0 || y == h - 1) ? 1.0 : 0.0;
			}
		}

		if (!snake.isEmpty()) {
			Position head = snake.getFirst();
			int headIdx = w * h * 2 + head.getY() * w + head.getX();
			grid[headIdx] = 1.0;

			for (int i = 1; i < snake.size(); i++) {
				Position p = snake.get(i);
				int bodyIdx = w * h + p.getY() * w + p.getX();
				grid[bodyIdx] = 1.0;
			}
		}

		if (food != null) {
			int foodIdx = w * h * 3 + food.getY() * w + food.getX();
			grid[foodIdx] = 1.0;
		}

		return new DQNState(w, h, grid, dir, terminal, score);
	}

	private static Direction inferDirection(List<Position> snakeHeadFirst) {
		if (snakeHeadFirst == null || snakeHeadFirst.size() < 2) {
			return Direction.RIGHT;
		}
		Position head = snakeHeadFirst.get(0);
		Position neck = snakeHeadFirst.get(1);
		int dx = head.getX() - neck.getX();
		int dy = head.getY() - neck.getY();

		if (dx == 1 && dy == 0) return Direction.RIGHT;
		if (dx == -1 && dy == 0) return Direction.LEFT;
		if (dx == 0 && dy == 1) return Direction.DOWN;
		if (dx == 0 && dy == -1) return Direction.UP;

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