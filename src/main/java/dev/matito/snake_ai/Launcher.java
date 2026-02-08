package dev.matito.snake_ai;

import dev.matito.snake_ai.ql.QLAgent;
import dev.matito.snake_ai.ql.State;

public class Launcher {
    public static void main(String[] args) {
        GameConfig config = new GameConfig("config.properties");

        if (config.isGuiEnabled()) {
            SnakeApplication.main(args);
        } else {
            runHeadless(config);
        }
    }

    private static void runHeadless(GameConfig config) {
        SnakeGame game = new SnakeGame(config.getGridWidth(), config.getGridHeight());
        QLAgent agent = new QLAgent();

        System.out.println("Running in headless mode");
        System.out.println("Grid size: " + config.getGridWidth() + "x" + config.getGridHeight());
        
        while (game.getGameState() != GameState.WON) {
            game.setDirection(agent.getAIDecision(State.fromGame(game)));
            game.step();
        }
        
        System.out.println("Final score: " + game.getScore());
        System.out.println("Game state: " + game.getGameState());
    }
}
