package dev.matito.snake_ai;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class Config {
    private final int gridWidth;
    private final int gridHeight;
    private final boolean guiEnabled;
    private final int gameSpeed;
    private final String agent;
    private final double agentAlpha;
    private final double agentGamma;
    private final int agentDqnReplay;
    private final int agentDqnBatch;
    private final boolean epsilonOverride;
    private final double agentEpsilonStart;
    private final double agentEpsilonMin;
    private final double epsilonDecay;

    public Config(String configPath) {
        Properties props = new Properties();
        try (InputStream input = new FileInputStream(configPath)) {
            props.load(input);
            this.gridWidth = Integer.parseInt(props.getProperty("grid.width", "20"));
            this.gridHeight = Integer.parseInt(props.getProperty("grid.height", "20"));
            this.guiEnabled = Boolean.parseBoolean(props.getProperty("gui.enabled", "true"));
            this.gameSpeed = Integer.parseInt(props.getProperty("game.speed", "150"));
            this.agent = props.getProperty("agent", "QL");
            this.agentAlpha = Double.parseDouble(props.getProperty("agent.alpha", "0.1"));
            this.agentGamma = Double.parseDouble(props.getProperty("agent.gamma", "0.9"));
            this.agentDqnReplay = Integer.parseInt(props.getProperty("agent.dqn.replay", "16"));
            this.agentDqnBatch = Integer.parseInt(props.getProperty("agent.dqn.batch", "8"));
            this.epsilonOverride = Boolean.parseBoolean(props.getProperty("agent.epsilon.override", "false"));
            this.agentEpsilonStart = Double.parseDouble(props.getProperty("agent.epsilon.start", "1.0"));
            this.agentEpsilonMin = Double.parseDouble(props.getProperty("agent.epsilon.min", "0.005"));
            this.epsilonDecay = Double.parseDouble(props.getProperty("agent.epsilon.decay", "0.9995"));
        } catch (IOException e) {
            throw new RuntimeException("Failed to load config: " + configPath, e);
        }
    }

    private void agentDqnReplay() {
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

    public double getAgentAlpha() {
        return agentAlpha;
    }

    public double getAgentGamma() {
        return agentGamma;
    }

    public int getAgentDqnReplay() {
        return agentDqnReplay;
    }

    public int getAgentDqnBatch() {
        return agentDqnBatch;
    }

    public boolean isEpsilonOverride() {
        return epsilonOverride;
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
