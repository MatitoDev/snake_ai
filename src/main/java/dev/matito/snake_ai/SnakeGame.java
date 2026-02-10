package dev.matito.snake_ai;

import dev.matito.snake_ai.ql.State;
import javafx.stage.Stage;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class SnakeGame {
    private final int gridWidth;
    private final int gridHeight;
    private final List<Position> snake;
    private Direction currentDirection;
    private Direction nextDirection;
    private Position food;
    private GameState gameState;
    private int score;
    private int lastScoreChange = 0;
    private boolean tooMuchSteps = false;
    private final Random random;
    private boolean ateFood;

    public SnakeGame(int gridWidth, int gridHeight) {
        this.gridWidth = gridWidth;
        this.gridHeight = gridHeight;
        this.snake = new ArrayList<>();
        this.random = new Random();
        this.score = 0;
        this.ateFood = false;
        reset();
    }

    public void reset() {
        snake.clear();
        int startX = gridWidth / 2;
        int startY = gridHeight / 2;
        snake.add(new Position(startX, startY));
        snake.add(new Position(startX - 1, startY));
        snake.add(new Position(startX - 2, startY));
        currentDirection = Direction.RIGHT;
        nextDirection = Direction.RIGHT;
        gameState = GameState.RUNNING;
        score = 0;
        tooMuchSteps = false;
        lastScoreChange = 0;
        ateFood = false;
        spawnFood();
    }

    public void setDirection(Direction direction) {
        if (direction != currentDirection.opposite()) {
            nextDirection = direction;
        }
    }

    public void step() {
        if (gameState != GameState.RUNNING) {
            return;
        }

        currentDirection = nextDirection;
        Position head = snake.getFirst();
        Position newHead = head.move(currentDirection);

        if (lastScoreChange > 200) tooMuchSteps = true;
        if (isOutOfBounds(newHead) || snake.contains(newHead) || tooMuchSteps) {
            gameState = GameState.GAME_OVER;
            return;
        }

        snake.add(0, newHead);

        if (newHead.equals(food)) {
            score++;
            lastScoreChange = 0;
            spawnFood();
            ateFood = true;
        } else {
            snake.remove(snake.size() - 1);
            ateFood = false;
            lastScoreChange++;
        }

        checkWinCondition();
    }

    private void checkWinCondition() {
        if (snake.size() == gridWidth * gridHeight) {
            gameState = GameState.WON;
            food = null;
        }
    }

    private boolean isOutOfBounds(Position pos) {
        return pos.getX() < 0 || pos.getX() >= gridWidth ||
                pos.getY() < 0 || pos.getY() >= gridHeight;
    }

    private void spawnFood() {
        Set<Position> occupiedPositions = new HashSet<>(snake);
        List<Position> availablePositions = new ArrayList<>();

        for (int x = 0; x < gridWidth; x++) {
            for (int y = 0; y < gridHeight; y++) {
                Position pos = new Position(x, y);
                if (!occupiedPositions.contains(pos)) {
                    availablePositions.add(pos);
                }
            }
        }

        if (availablePositions.isEmpty()) {
            checkWinCondition();
            return;
        }

        food = availablePositions.get(random.nextInt(availablePositions.size()));
    }

    public void setFoodPosition(Position foodPosition) {
        this.food = foodPosition;
    }

    public List<Position> getSnakePositions() {
        return new ArrayList<>(snake);
    }

    public Position getFoodPosition() {
        return food;
    }

    public GameState getGameState() {
        return gameState;
    }

    public int getScore() {
        return score;
    }

    public int getGridWidth() {
        return gridWidth;
    }

    public int getGridHeight() {
        return gridHeight;
    }

    public State getState() {
        return State.of(
                gridWidth,
                gridHeight,
                snake,
                currentDirection,
                food,
                gameState == GameState.GAME_OVER,
                score
        );
    }

    public boolean isTooMuchSteps() {
        return tooMuchSteps;
    }
}
