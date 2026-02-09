package dev.matito.snake_ai;

import dev.matito.snake_ai.dqn.DQNAgent;
import dev.matito.snake_ai.ql.QLAgent;
import dev.matito.snake_ai.ql.State;
import javafx.animation.AnimationTimer;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.stage.Stage;

import java.util.List;

public class SnakeGUI {
    private final SnakeGame game;
    private final Canvas canvas;
    private final int cellSize;
    private final GraphicsContext gc;
    private AnimationTimer gameLoop;
    private long lastUpdate;
    private final int gameSpeed;

    public SnakeGUI(SnakeGame game, int gameSpeed) {
        this.game = game;
        this.gameSpeed = gameSpeed;
        this.cellSize = 20;
        this.canvas = new Canvas(game.getGridWidth() * cellSize, game.getGridHeight() * cellSize);
        this.gc = canvas.getGraphicsContext2D();
        this.lastUpdate = 0;
        setupKeyboardControls();
        render();
    }

    private void setupKeyboardControls() {
        canvas.setFocusTraversable(true);
        canvas.setOnKeyPressed(this::handleKeyPress);
    }

    private void handleKeyPress(KeyEvent event) {
        KeyCode code = event.getCode();
        switch (code) {
            case W, UP -> game.setDirection(Direction.UP);
            case S, DOWN -> game.setDirection(Direction.DOWN);
            case A, LEFT -> game.setDirection(Direction.LEFT);
            case D, RIGHT -> game.setDirection(Direction.RIGHT);
            case R -> {
                if (game.getGameState() == GameState.GAME_OVER) {
                    game.reset();
                }
            }
            //case L -> game.step();
        }
    }

    public void startQL() {
        QLAgent agent = new QLAgent();
        gameLoop = new AnimationTimer() {
            @Override
            public void handle(long now) {
                if (now - lastUpdate >= gameSpeed * 1_000_000L) {
                    game.setDirection(agent.getAIDecision(State.fromGame(game)));
                    if (game.getGameState() == GameState.GAME_OVER) {
                        System.out.println(game.getScore() + " - " + agent.getEpsilon());
                        game.reset();
                    }
                    game.step();
                    render();
                    lastUpdate = now;
                }
            }
        };
        gameLoop.start();
    }

    public void startDQN() {
        DQNAgent agent = new DQNAgent(game.getGridWidth(), game.getGridHeight(), 32, 123456789L);
        agent.resetEpisode();
        gameLoop = new AnimationTimer() {
            @Override
            public void handle(long now) {
                if (now - lastUpdate >= gameSpeed * 1_000_000L) {
                    game.setDirection(agent.decide(game));
                    if (game.getGameState() == GameState.GAME_OVER) {
                        System.out.println(game.getScore() + " - " + agent.getEpsilon());
                        game.reset();
                    }
                    agent.train(8);
                    game.step();
                    render();
                    lastUpdate = now;
                }
            }
        };
        gameLoop.start();
    }

    public void stop() {
        if (gameLoop != null) {
            gameLoop.stop();
        }
    }

    private void render() {
        gc.setFill(Color.BLACK);
        gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());

        gc.setFill(Color.RED);
        Position food = game.getFoodPosition();
        gc.fillRect(food.getX() * cellSize, food.getY() * cellSize, cellSize, cellSize);

        gc.setFill(Color.GREEN);
        List<Position> snakePositions = game.getSnakePositions();
        for (int i = 0; i < snakePositions.size(); i++) {
            Position pos = snakePositions.get(i);
            gc.fillRect(pos.getX() * cellSize, pos.getY() * cellSize, cellSize, cellSize);

            if (i == 0) {
                gc.setFill(Color.BLUE);
                double eyeSize = cellSize / 5.0;
                double x = pos.getX() * cellSize;
                double y = pos.getY() * cellSize;

                gc.fillOval(x + cellSize * 0.3, y + cellSize * 0.3, eyeSize, eyeSize);
                gc.fillOval(x + cellSize * 0.6, y + cellSize * 0.3, eyeSize, eyeSize);
                gc.setFill(Color.GREEN);
            }
        }

        gc.setFill(Color.WHITE);
        gc.setFont(new Font(16));
        gc.fillText("Score: " + game.getScore(), 10, 20);

        if (game.getGameState() == GameState.GAME_OVER) {
            gc.setFill(Color.RED);
            gc.setFont(new Font(32));
            gc.fillText("GAME OVER", canvas.getWidth() / 2 - 100, canvas.getHeight() / 2);
            gc.setFont(new Font(16));
            gc.fillText("Press R to restart", canvas.getWidth() / 2 - 70, canvas.getHeight() / 2 + 30);
        }
    }



    public Canvas getCanvas() {
        return canvas;
    }
}
