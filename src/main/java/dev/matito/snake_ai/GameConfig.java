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

    public GameConfig(String configPath) {
        Properties props = new Properties();
        try (InputStream input = new FileInputStream(configPath)) {
            props.load(input);
            this.gridWidth = Integer.parseInt(props.getProperty("grid.width", "20"));
            this.gridHeight = Integer.parseInt(props.getProperty("grid.height", "20"));
            this.guiEnabled = Boolean.parseBoolean(props.getProperty("gui.enabled", "true"));
            this.gameSpeed = Integer.parseInt(props.getProperty("game.speed", "150"));
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
}
