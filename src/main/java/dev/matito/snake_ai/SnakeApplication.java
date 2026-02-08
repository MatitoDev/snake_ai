package dev.matito.snake_ai;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;

public class SnakeApplication extends Application {
    private SnakeGUI snakeGUI;

    @Override
    public void start(Stage stage) {
        GameConfig config = new GameConfig("config.properties");
        SnakeGame game = new SnakeGame(config.getGridWidth(), config.getGridHeight());
        snakeGUI = new SnakeGUI(game, config.getGameSpeed());

        StackPane root = new StackPane(snakeGUI.getCanvas());
        Scene scene = new Scene(root);

        stage.setTitle("Snake Game");
        stage.setScene(scene);
        stage.setResizable(false);
        stage.show();


        snakeGUI.getCanvas().requestFocus();
        snakeGUI.start();
    }

    @Override
    public void stop() {
        if (snakeGUI != null) {
            snakeGUI.stop();
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}
