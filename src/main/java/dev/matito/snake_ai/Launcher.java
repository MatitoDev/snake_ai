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
        DataLogger logger = new DataLogger("ql_training_data");

        System.out.println("Running in headless QL mode");
        System.out.println("Grid size: " + config.getGridWidth() + "x" + config.getGridHeight());
        System.out.println("Data will auto-save on Ctrl+C interrupt");
        int i = 1;
        while (!(game.getGameState() == GameState.WON || agent.getEpsilon() <= agent.getEpsilonMin())) {
            Direction chosenDirection = agent.getAIDecision(State.fromGame(game));
            game.setDirection(chosenDirection);
            game.step();
            stats.onStep(game.getScore());
            
            if (game.getGameState() == GameState.GAME_OVER) {
                System.out.println(game.getScore() + " - " + agent.getEpsilon() + " - " + i);
                stats.onEpisodeEnd(game.getScore());
                logger.logEpisodeEnd(agent.getEpsilon(), game.getScore(), false, config.getGridWidth(), config.getGridHeight());
                game.reset();
                i++;
            }
        }

        System.out.println("Try to get Highscore");
        int highscore = 0;
        int steps = 0;
        while (game.getGameState() != GameState.WON) {
            agent.setExploration(0.0, 0.0, 0.0);
            Direction chosenDirection = agent.decide(game);
            game.setDirection(chosenDirection);
            game.step();
            steps++;
            stats.onStep(game.getScore());
            logger.logStep(agent.getLastReward(), game.getScore(), chosenDirection, game.getFoodPosition());
            
            if (game.getGameState() == GameState.GAME_OVER) {
                if (game.getScore() > highscore) {
                    System.out.println("New Highscore: " + game.getScore());
                    System.out.println("Steps: " + steps);
                    steps = 0;
                    stats.onEpisodeEnd(game.getScore());
                    logger.logEpisodeEnd(0.0, game.getScore(), true,
                        config.getGridWidth(), config.getGridHeight());
                    highscore = game.getScore();
                }
                game.reset();
            }
        }

        System.out.println("Final score: " + game.getScore());
        System.out.println("Game state: " + game.getGameState());
        
        try {
            logger.save("ql_training_data");
            System.out.println("Training data saved to ql_training_data_*.csv");
        } catch (java.io.IOException e) {
            System.err.println("Failed to save training data: " + e.getMessage());
        }
    }

    private static void runHeadlessDQN(Config config) {
        SnakeGame game = new SnakeGame(config.getGridWidth(), config.getGridHeight());
        DQNAgent agent = new DQNAgent(config.getGridWidth(), config.getGridHeight(), config.getAgentDqnReplay(), 123456789L);
        agent.setLearningRate(config.getAgentAlpha());
        agent.setGamma(config.getAgentGamma());
        agent.resetEpisode();

        TrainingStats stats = new TrainingStats();
        DashboardServer dash = DashboardServer.start("config.properties", agent, stats);
        DataLogger logger = new DataLogger("dqn_training_data");

        System.out.println("Running in headless DQN mode");
        System.out.println("Grid size: " + config.getGridWidth() + "x" + config.getGridHeight());
        System.out.println("Data will auto-save on Ctrl+C interrupt");
        int i = 1;
        while (!(game.getGameState() == GameState.WON || agent.getEpsilon() <= agent.getEpsilonMin())) {
            Direction chosenDirection = agent.decide(game);
            game.setDirection(chosenDirection);
            agent.train(config.getAgentDqnBatch());
            game.step();
            stats.onStep(game.getScore());
            
            if (game.getGameState() == GameState.GAME_OVER) {
                System.out.println(game.getScore() + " - " + agent.getEpsilon() + " - " + i);
                stats.onEpisodeEnd(game.getScore());
                logger.logEpisodeEnd(agent.getEpsilon(), game.getScore(), false,
                    config.getGridWidth(), config.getGridHeight());
                game.reset();
                agent.resetEpisode();
                i++;
            }
        }

        System.out.println("Try to get Highscore");
        int highscore = 0;
        while (game.getGameState() != GameState.WON) {
            agent.setEpsilon(0.0, 0.0, 0.0);
            Direction chosenDirection = agent.decide(game);
            game.setDirection(chosenDirection);
            game.step();
            stats.onStep(game.getScore());
            logger.logStep(agent.getLastReward(), game.getScore(), chosenDirection, game.getFoodPosition());
            
            if (game.getGameState() == GameState.GAME_OVER) {
                if (game.getScore() > highscore) {
                    System.out.println("New Highscore: " + game.getScore());
                    stats.onEpisodeEnd(game.getScore());
                    logger.logEpisodeEnd(0.0, game.getScore(), true,
                        config.getGridWidth(), config.getGridHeight());
                    highscore = game.getScore();
                }
                game.reset();
            }
        }

        System.out.println("Final score: " + game.getScore());
        System.out.println("Game state: " + game.getGameState());
        
        try {
            logger.save("dqn_training_data");
            System.out.println("Training data saved to dqn_training_data_*.csv");
        } catch (java.io.IOException e) {
            System.err.println("Failed to save training data: " + e.getMessage());
        }
    }
}
