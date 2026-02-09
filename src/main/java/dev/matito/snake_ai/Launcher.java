package dev.matito.snake_ai;

import dev.matito.snake_ai.dqn.DQNAgent;
import dev.matito.snake_ai.ql.QLAgent;
import dev.matito.snake_ai.ql.State;

import java.util.Objects;

public class Launcher {
    public static void main(String[] args) {
        Config config = new Config("config.properties");

        if (config.isGuiEnabled()) SnakeApplication.main(args);
        else if (Objects.equals(config.getAgent(), "QL")) runHeadlessQL(config);
            else if (Objects.equals(config.getAgent(), "DQN")) runHeadlessDQN(config);
    }

    private static void runHeadlessQL(Config config) {
        SnakeGame game = new SnakeGame(config.getGridWidth(), config.getGridHeight());
        QLAgent agent = new QLAgent();
        agent.setLearning(config.getAgentAlpha(), config.getAgentGamma());

        System.out.println("Running in headless QL mode");
        System.out.println("Grid size: " + config.getGridWidth() + "x" + config.getGridHeight());
        int i = 1;
        while (game.getGameState() != GameState.WON || agent.getEpsilon() == config.getAgentEpsilonMin()) {
            game.setDirection(agent.getAIDecision(State.fromGame(game)));
            if (game.getGameState() == GameState.GAME_OVER) {
                System.out.println(game.getScore() + " - " + agent.getEpsilon() + " - " + i);
                game.reset();
                i++;
            }
            game.step();
        }

        System.out.println("Final score: " + game.getScore());
        System.out.println("Game state: " + game.getGameState());
    }

    private static void runHeadlessDQN(Config config) {
        SnakeGame game = new SnakeGame(config.getGridWidth(), config.getGridHeight());
        DQNAgent agent = new DQNAgent(config.getGridWidth(), config.getGridHeight(), config.getAgentDqnReplay(), 123456789L);
        agent.setLearningRate(config.getAgentAlpha());
        agent.setGamma(config.getAgentGamma());
        agent.resetEpisode();

        System.out.println("Running in headless DQN mode");
        System.out.println("Grid size: " + config.getGridWidth() + "x" + config.getGridHeight());
        int i = 1;
        while (game.getGameState() != GameState.WON || agent.getEpsilon() == config.getAgentEpsilonMin()) {
            game.setDirection(agent.decide(game));
            if (game.getGameState() == GameState.GAME_OVER) {
                System.out.println(game.getScore() + " - " + agent.getEpsilon() + " - " + i);
                game.reset();
                i++;
            }
            agent.train(config.getAgentDqnBatch());
            game.step();
        }

        System.out.println("Final score: " + game.getScore());
        System.out.println("Game state: " + game.getGameState());
    }
}
