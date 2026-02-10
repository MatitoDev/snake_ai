package dev.matito.snake_ai;

import java.io.*;
import java.nio.file.*;
import java.util.*;

public final class DataLogger implements AutoCloseable {
    
    private final List<EpisodeData> episodes = new ArrayList<>();
    private final List<StepData> currentEpisodeSteps = new ArrayList<>();
    private ReplayData bestReplay = null;
    
    private int bestScore = 0;
    private int currentEpisode = 0;
    private String baseFilename = "training_data";
    private volatile boolean shutdownHookInstalled = false;
    
    public DataLogger() {
        installShutdownHook();
    }
    
    public DataLogger(String baseFilename) {
        this.baseFilename = baseFilename;
        installShutdownHook();
    }
    
    private void installShutdownHook() {
        if (shutdownHookInstalled) return;
        shutdownHookInstalled = true;
        
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                System.out.println("\nInterrupted - saving training data...");
                save(baseFilename);
                System.out.println("Training data saved successfully!");
            } catch (IOException e) {
                System.err.println("Failed to save training data on shutdown: " + e.getMessage());
            }
        }, "DataLogger-shutdown"));
    }
    
    public void logStep(double reward, int score, Direction direction, Position foodPosition) {
        // Only track if we're doing well enough that this might be the best
        if (currentEpisodeSteps.size() < 10000) { // Safety limit
            currentEpisodeSteps.add(new StepData(reward, score, direction, foodPosition));
        }
    }
    
    public void logEpisodeEnd(double epsilon, int score, boolean isGreedy, int gridWidth, int gridHeight) {
        currentEpisode++;
        
        int highscore = isGreedy ? score : 0;
        episodes.add(new EpisodeData(currentEpisode, epsilon, score, highscore));
        
        // Only save if this is actually better
        if (score > bestScore && !currentEpisodeSteps.isEmpty()) {
            bestScore = score;
            bestReplay = new ReplayData(gridWidth, gridHeight, new ArrayList<>(currentEpisodeSteps));
            System.out.println("New Highscore: " + bestScore);
        }
        
        currentEpisodeSteps.clear();
    }
    
    public void save(String baseFilename) throws IOException {
        saveEpisodes(baseFilename + "_episodes.csv");
        saveSteps(baseFilename + "_steps.csv");
        if (bestReplay != null) {
            saveBestReplay(baseFilename + "_best_replay.txt");
        }
    }
    
    private void saveEpisodes(String filename) throws IOException {
        Path path = Paths.get(filename);
        try (BufferedWriter writer = Files.newBufferedWriter(path)) {
            writer.write("try,epsilon,score,highscore\n");
            for (EpisodeData ep : episodes) {
                writer.write(String.format(Locale.US, "%d,%.6f,%d,%d\n",
                    ep.tryNumber, ep.epsilon, ep.score, ep.highscore));
            }
        }
    }
    
    private void saveSteps(String filename) throws IOException {
        Path path = Paths.get(filename);
        try (BufferedWriter writer = Files.newBufferedWriter(path)) {
            writer.write("step,reward\n");
            int stepCounter = 0;
            for (EpisodeData ep : episodes) {
                for (int i = 0; i < ep.score; i++) {
                    stepCounter++;
                    writer.write(String.format(Locale.US, "%d,0.0\n", stepCounter));
                }
            }
        }
    }
    
    private void saveBestReplay(String filename) throws IOException {
        Path path = Paths.get(filename);
        try (BufferedWriter writer = Files.newBufferedWriter(path)) {
            writer.write("# Best replay - Score: " + bestScore + "\n");
            writer.write("# Grid: " + bestReplay.gridWidth + "x" + bestReplay.gridHeight + "\n");
            writer.write("# Format: DIRECTION,FOOD_X,FOOD_Y,REWARD,SCORE\n");
            writer.write(bestReplay.gridWidth + "," + bestReplay.gridHeight + "\n");
            
            for (StepData step : bestReplay.steps) {
                writer.write(String.format(Locale.US, "%s,%d,%d,%.6f,%d\n",
                    step.direction.name(),
                    step.foodPosition.getX(),
                    step.foodPosition.getY(),
                    step.reward,
                    step.score));
            }
        }
    }
    
    @Override
    public void close() throws IOException {
        save("training_data");
    }
    
    private static final class EpisodeData {
        final int tryNumber;
        final double epsilon;
        final int score;
        final int highscore;
        
        EpisodeData(int tryNumber, double epsilon, int score, int highscore) {
            this.tryNumber = tryNumber;
            this.epsilon = epsilon;
            this.score = score;
            this.highscore = highscore;
        }
    }
    
    private static final class StepData {
        final double reward;
        final int score;
        final Direction direction;
        final Position foodPosition;
        
        StepData(double reward, int score, Direction direction, Position foodPosition) {
            this.reward = reward;
            this.score = score;
            this.direction = direction;
            this.foodPosition = foodPosition;
        }
    }
    
    private static final class ReplayData {
        final int gridWidth;
        final int gridHeight;
        final List<StepData> steps;
        
        ReplayData(int gridWidth, int gridHeight, List<StepData> steps) {
            this.gridWidth = gridWidth;
            this.gridHeight = gridHeight;
            this.steps = steps;
        }
    }
}
