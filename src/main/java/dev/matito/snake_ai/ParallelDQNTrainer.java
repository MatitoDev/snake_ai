package dev.matito.snake_ai;

import dev.matito.snake_ai.dqn.DQNAgent;
import dev.matito.snake_ai.dashboard.TrainingStats;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * Parallel DQN trainer that runs multiple games simultaneously.
 * Each game collects experiences independently, all feed into shared replay buffer.
 */
public class ParallelDQNTrainer {
    
    private final Config config;
    private final DQNAgent agent;
    private final TrainingStats stats;
    private final int numParallelGames;
    private final ExecutorService executor;
    
    public ParallelDQNTrainer(Config config, DQNAgent agent, TrainingStats stats, int numParallelGames) {
        this.config = config;
        this.agent = agent;
        this.stats = stats;
        this.numParallelGames = numParallelGames;
        this.executor = Executors.newFixedThreadPool(numParallelGames + 1); // +1 for training thread
    }
    
    public void train() {
        System.out.println("Running parallel DQN with " + numParallelGames + " games");
        
        List<Future<GameWorkerResult>> futures = new ArrayList<>();
        
        // Start parallel game workers
        for (int workerId = 0; workerId < numParallelGames; workerId++) {
            GameWorker worker = new GameWorker(workerId);
            futures.add(executor.submit(worker));
        }
        
        // Separate training thread
        Thread trainingThread = new Thread(this::trainingLoop, "dqn-trainer");
        trainingThread.start();
        
        // Wait for all workers to complete
        int totalEpisodes = 0;
        for (Future<GameWorkerResult> future : futures) {
            try {
                GameWorkerResult result = future.get();
                totalEpisodes += result.episodesCompleted;
                System.out.println("Worker " + result.workerId + " completed " + result.episodesCompleted + " episodes");
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }
        
        trainingThread.interrupt();
        try {
            trainingThread.join(5000); // Wait up to 5 seconds for training thread to finish
        } catch (InterruptedException e) {
            // Ignore
        }
        
        executor.shutdown();
        try {
            if (!executor.awaitTermination(10, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }
        
        System.out.println("Parallel training complete: " + totalEpisodes + " total episodes");
    }
    
    private void trainingLoop() {
        int trainCount = 0;
        int trainFreq = config.getAgentDqnTrainFreq();
        int stepsSinceLastTrain = 0;
        
        while (!Thread.currentThread().isInterrupted() && agent.getEpsilon() > agent.getEpsilonMin()) {
            if (agent.getReplayBufferSize() >= config.getAgentDqnBatch()) {
                stepsSinceLastTrain++;
                
                if (stepsSinceLastTrain >= trainFreq) {
                    agent.train(config.getAgentDqnBatch());
                    trainCount++;
                    stepsSinceLastTrain = 0;
                    
                    if (trainCount % 1000 == 0) {
                        System.out.println("Training updates: " + trainCount + ", epsilon: " + 
                            String.format("%.4f", agent.getEpsilon()) + ", buffer: " + agent.getReplayBufferSize());
                    }
                }
            } else {
                try {
                    Thread.sleep(10); // Wait for buffer to fill
                } catch (InterruptedException e) {
                    break;
                }
            }
        }
        System.out.println("Training thread finished: " + trainCount + " updates");
    }
    
    private class GameWorker implements Callable<GameWorkerResult> {
        private final int workerId;
        
        GameWorker(int workerId) {
            this.workerId = workerId;
        }
        
        @Override
        public GameWorkerResult call() {
            SnakeGame game = new SnakeGame(config.getGridWidth(), config.getGridHeight());
            int episodes = 0;
            
            while (agent.getEpsilon() > agent.getEpsilonMin() && !Thread.currentThread().isInterrupted()) {
                // Collect experience
                Direction direction = agent.decide(game);
                game.setDirection(direction);
                game.step();
                
                if (game.getGameState() == GameState.GAME_OVER) {
                    stats.onEpisodeEnd(game.getScore());
                    episodes++;
                    
                    if (episodes % 100 == 0) {
                        System.out.println("Worker " + workerId + ": " + episodes + 
                            " episodes, score: " + game.getScore() + 
                            ", epsilon: " + String.format("%.4f", agent.getEpsilon()));
                    }
                    
                    game.reset();
                    agent.resetEpisode();
                }
            }
            
            return new GameWorkerResult(workerId, episodes);
        }
    }
    
    private static class GameWorkerResult {
        final int workerId;
        final int episodesCompleted;
        
        GameWorkerResult(int workerId, int episodesCompleted) {
            this.workerId = workerId;
            this.episodesCompleted = episodesCompleted;
        }
    }
}
