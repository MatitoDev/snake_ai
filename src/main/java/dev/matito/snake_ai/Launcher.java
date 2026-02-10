package dev.matito.snake_ai;

import dev.matito.snake_ai.dashboard.DashboardServer;
import dev.matito.snake_ai.dashboard.TrainingStats;
import dev.matito.snake_ai.dqn.DQNAgent;
import dev.matito.snake_ai.ql.QLAgent;
import dev.matito.snake_ai.ql.State;

import java.util.Objects;

public class Launcher {
    public static <DashboardServer> void main(String[] args) {
        Config config = new Config("config.properties");

        if (config.isGuiEnabled()) SnakeApplication.main(args);
        else if (Objects.equals(config.getAgent(), "QL")) runHeadlessQL(config);
            else if (Objects.equals(config.getAgent(), "DQN")) runHeadlessDQN(config);
    }

    private static void runHeadlessQL(Config config) {
        SnakeGame game = new SnakeGame(config.getGridWidth(), config.getGridHeight());
        QLAgent agent = new QLAgent();
        agent.setLearning(config.getAgentAlpha(), config.getAgentGamma());

        TrainingStats stats = new TrainingStats();
        DashboardServer dash = DashboardServer.start("config.properties", agent, stats);

        System.out.println("Running in headless QL mode");
        System.out.println("Grid size: " + config.getGridWidth() + "x" + config.getGridHeight());
        int i = 1;
        while (!(game.getGameState() == GameState.WON || agent.getEpsilon() <= agent.getEpsilonMin())) {
            game.setDirection(agent.getAIDecision(State.fromGame(game)));
            if (game.getGameState() == GameState.GAME_OVER) {
                //System.out.println(agent.getEpsilonDecay());
                System.out.println(game.getScore() + " - " + agent.getEpsilon() + " - " + i);
                stats.onEpisodeEnd(game.getScore());
                game.reset();
                i++;
            }
            game.step();
            stats.onStep(game.getScore());
        }

        System.out.println("Try to get Highscore");
        int highscore = 0;
        int steps = 0;
        while (game.getGameState() != GameState.WON) {
            agent.setExploration(0.0, 0.0, 0.0);
            game.setDirection(agent.decide(game));
            if (game.getGameState() == GameState.GAME_OVER) {
                if (game.getScore() > highscore) {
                    System.out.println("New Highscore: " + game.getScore());
                    System.out.println("Steps: " + steps);
                    steps = 0;
                    stats.onEpisodeEnd(game.getScore());
                    highscore = game.getScore();
                }
                game.reset();
            }
            game.step();
            steps++;
            stats.onStep(game.getScore());
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

        TrainingStats stats = new TrainingStats();
        DashboardServer dash = DashboardServer.start("config.properties", agent, stats);

        System.out.println("Running in headless DQN mode");
        System.out.println("Grid size: " + config.getGridWidth() + "x" + config.getGridHeight());
        int i = 1;
        while (!(game.getGameState() == GameState.WON || agent.getEpsilon() <= agent.getEpsilonMin())) {
            game.setDirection(agent.decide(game));
            if (game.getGameState() == GameState.GAME_OVER) {
                System.out.println(game.getScore() + " - " + agent.getEpsilon() + " - " + i);
                stats.onEpisodeEnd(game.getScore());
                game.reset();
                agent.resetEpisode();
                i++;
            }
            agent.train(config.getAgentDqnBatch());
            game.step();
            stats.onStep(game.getScore());
        }

        System.out.println("Try to get Highscore");
        int highscore = 0;
        while (game.getGameState() != GameState.WON) {
            agent.setEpsilon(0.0, 0.0, 0.0);
            game.setDirection(agent.decide(game));
            if (game.getGameState() == GameState.GAME_OVER) {
                if (game.getScore() > highscore) {
                    System.out.println("New Highscore: " + game.getScore());
                    stats.onEpisodeEnd(game.getScore());
                    highscore = game.getScore();
                }
                game.reset();
            }
            game.step();
            stats.onStep(game.getScore());
        }

        System.out.println("Final score: " + game.getScore());
        System.out.println("Game state: " + game.getGameState());
    }
}
