package dev.matito.snake_ai;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class GameConfig {
    private final int gridWidth;
    private final int gridHeight;
    private final boolean guiEnabled;
    private final int gameSpeed;
    private final String agent;
    private final double agentEpsilonStart;
    private final double agentEpsilonMin;
    private final double epsilonDecay;

    public GameConfig(String configPath) {
        Properties props = new Properties();
        try (InputStream input = new FileInputStream(configPath)) {
            props.load(input);
            this.gridWidth = Integer.parseInt(props.getProperty("grid.width", "20"));
            this.gridHeight = Integer.parseInt(props.getProperty("grid.height", "20"));
            this.guiEnabled = Boolean.parseBoolean(props.getProperty("gui.enabled", "true"));
            this.gameSpeed = Integer.parseInt(props.getProperty("game.speed", "150"));
            this.agent = props.getProperty("agent", "QL");
            this.agentEpsilonStart = Double.parseDouble(props.getProperty("agent.epsilon.start", "1.0"));
            this.agentEpsilonMin = Double.parseDouble(props.getProperty("agent.epsilon.min", "0.005"));
            this.epsilonDecay = Double.parseDouble(props.getProperty("agent.epsilon.decay", "0.9995"));
        } catch (IOException e) {
            throw new RuntimeException("Failed to load config: " + configPath, e);
        }
    }

    public int getGridWidth() {
        return gridWidth;
    }

    public int getGridHeight() {
        return gridHeight;
    }

    public boolean isGuiEnabled() {
        return guiEnabled;
    }

    public int getGameSpeed() {
        return gameSpeed;
    }

    public double getAgentEpsilonMin() {
        return agentEpsilonMin;
    }

    public double getAgentEpsilonStart() {
        return agentEpsilonStart;
    }

    public double getEpsilonDecay() {
        return epsilonDecay;
    }

    public String getAgent() {
        return agent;
    }
}
