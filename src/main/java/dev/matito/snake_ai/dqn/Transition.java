package dev.matito.snake_ai.dqn;

public final class Transition {
	private final double[] state;
	private final int action;
	private final double reward;
	private final double[] nextState;
	private final boolean terminal;

	public Transition(double[] state, int action, double reward, double[] nextState, boolean terminal) {
		this.state = state;
		this.action = action;
		this.reward = reward;
		this.nextState = nextState;
		this.terminal = terminal;
	}

	public double[] getState() {
		return state;
	}

	public int getAction() {
		return action;
	}

	public double getReward() {
		return reward;
	}

	public double[] getNextState() {
		return nextState;
	}

	public boolean isTerminal() {
		return terminal;
	}
}