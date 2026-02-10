package dev.matito.snake_ai.ql;

import dev.matito.snake_ai.Direction;
import dev.matito.snake_ai.GameState;
import dev.matito.snake_ai.Position;
import dev.matito.snake_ai.SnakeGame;

import java.util.BitSet;
import java.util.List;

/**
 * Immutable snapshot of the board for the Q learning agent.
 * Snake positions are expected head first.
 */
public final class State {
	private final int width;
	private final int height;
	private final Position[] snake;
	private final Direction direction;
	private final Position food;
	private final boolean terminal;
	private final int score;
	private final BitSet occupied;

	private State(
			int width,
			int height,
			List<Position> snakeHeadFirst,
			Direction direction,
			Position food,
			boolean terminal,
			int score
	) {
		if (width <= 0 || height <= 0) {
			throw new IllegalArgumentException("Invalid board size");
		}
		if (snakeHeadFirst == null || snakeHeadFirst.isEmpty()) {
			throw new IllegalArgumentException("Snake is null or empty");
		}
		if (direction == null) {
			throw new IllegalArgumentException("Direction is null");
		}
		if (food == null) {
			throw new IllegalArgumentException("Food is null");
		}

		this.width = width;
		this.height = height;
		this.direction = direction;
		this.food = food;
		this.terminal = terminal;
		this.score = score;

		this.snake = snakeHeadFirst.toArray(new Position[0]);
		this.occupied = new BitSet(width * height);

		for (int i = 0; i < snake.length; i++) {
			Position p = snake[i];
			if (p == null) {
				throw new IllegalArgumentException("Snake contains null at index " + i);
			}
			if (!inBounds(p.getX(), p.getY())) {
				throw new IllegalArgumentException("Snake out of bounds at index " + i);
			}
			occupied.set(index(p.getX(), p.getY()));
		}

		if (!inBounds(food.getX(), food.getY())) {
			throw new IllegalArgumentException("Food out of bounds");
		}
	}

	/** Creates a state snapshot. */
	public static State of(
			int width,
			int height,
			List<Position> snakeHeadFirst,
			Direction direction,
			Position food,
			boolean terminal,
			int score
	) {
		return new State(width, height, snakeHeadFirst, direction, food, terminal, score);
	}

	/** Creates a state snapshot from {@link SnakeGame}. Direction is inferred from head and neck. */
	public static State fromGame(SnakeGame game) {
		if (game == null) {
			throw new IllegalArgumentException("Game is null");
		}

		List<Position> snake = game.getSnakePositionsDirect();
		Direction dir = inferDirection(snake);
		boolean terminal = game.getGameState() != GameState.RUNNING;

		return new State(
				game.getGridWidth(),
				game.getGridHeight(),
				snake,
				dir,
				game.getFoodPosition(),
				terminal,
				game.getScore()
		);
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

	public int getWidth() {
		return width;
	}

	public int getHeight() {
		return height;
	}

	public int getLength() {
		return snake.length;
	}

	public Direction getDirection() {
		return direction;
	}

	public Position getFood() {
		return food;
	}

	public boolean isTerminal() {
		return terminal;
	}

	public int getScore() {
		return score;
	}

	public Position getHead() {
		return snake[0];
	}

	public Position getTail() {
		return snake[snake.length - 1];
	}

	public int getHeadX() {
		return snake[0].getX();
	}

	public int getHeadY() {
		return snake[0].getY();
	}

	public int getTailX() {
		return snake[snake.length - 1].getX();
	}

	public int getTailY() {
		return snake[snake.length - 1].getY();
	}

	public Position getSnakeAt(int index) {
		if (index < 0 || index >= snake.length) {
			throw new IndexOutOfBoundsException("index=" + index);
		}
		return snake[index];
	}

	public boolean inBounds(int x, int y) {
		return x >= 0 && y >= 0 && x < width && y < height;
	}

	public boolean isOccupied(int x, int y) {
		if (!inBounds(x, y)) {
			return false;
		}
		return occupied.get(index(x, y));
	}

	private int index(int x, int y) {
		return y * width + x;
	}
}
