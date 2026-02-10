package dev.matito.snake_ai;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Replays a recorded game from a replay file.
 * The replay file contains grid dimensions and a sequence of moves with food positions.
 */
public final class ReplayPlayer {
    
    private final int gridWidth;
    private final int gridHeight;
    private final List<ReplayStep> steps;
    private int currentStep = 0;
    
    private ReplayPlayer(int gridWidth, int gridHeight, List<ReplayStep> steps) {
        this.gridWidth = gridWidth;
        this.gridHeight = gridHeight;
        this.steps = steps;
    }
    
    /**
     * Loads a replay from file.
     * Expected format:
     * Line 1: gridWidth,gridHeight
     * Following lines: DIRECTION,FOOD_X,FOOD_Y,REWARD,SCORE
     */
    public static ReplayPlayer load(String filename) throws IOException {
        Path path = Paths.get(filename);
        List<ReplayStep> steps = new ArrayList<>();
        int gridWidth = 0;
        int gridHeight = 0;
        
        try (BufferedReader reader = Files.newBufferedReader(path)) {
            String line;
            boolean firstDataLine = true;
            
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                
                String[] parts = line.split(",");
                
                if (firstDataLine) {
                    gridWidth = Integer.parseInt(parts[0]);
                    gridHeight = Integer.parseInt(parts[1]);
                    firstDataLine = false;
                } else if (parts.length >= 5) {
                    Direction direction = Direction.valueOf(parts[0].trim());
                    int foodX = Integer.parseInt(parts[1].trim());
                    int foodY = Integer.parseInt(parts[2].trim());
                    double reward = Double.parseDouble(parts[3].trim());
                    int score = Integer.parseInt(parts[4].trim());
                    
                    steps.add(new ReplayStep(direction, new Position(foodX, foodY), reward, score));
                }
            }
        }
        
        return new ReplayPlayer(gridWidth, gridHeight, steps);
    }
    
    public int getGridWidth() {
        return gridWidth;
    }
    
    public int getGridHeight() {
        return gridHeight;
    }
    
    public boolean hasNextStep() {
        return currentStep < steps.size();
    }
    
    public ReplayStep getNextStep() {
        if (!hasNextStep()) {
            throw new IllegalStateException("No more replay steps");
        }
        return steps.get(currentStep++);
    }
    
    public int getCurrentStepNumber() {
        return currentStep;
    }
    
    public int getTotalSteps() {
        return steps.size();
    }
    
    public void reset() {
        currentStep = 0;
    }
    
    public static final class ReplayStep {
        private final Direction direction;
        private final Position foodPosition;
        private final double reward;
        private final int score;
        
        ReplayStep(Direction direction, Position foodPosition, double reward, int score) {
            this.direction = direction;
            this.foodPosition = foodPosition;
            this.reward = reward;
            this.score = score;
        }
        
        public Direction getDirection() {
            return direction;
        }
        
        public Position getFoodPosition() {
            return foodPosition;
        }
        
        public double getReward() {
            return reward;
        }
        
        public int getScore() {
            return score;
        }
    }
}
