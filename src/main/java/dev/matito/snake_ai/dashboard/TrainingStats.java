package dev.matito.snake_ai.dashboard;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;

public final class TrainingStats implements AutoCloseable {

	private final LongAdder steps = new LongAdder();
	private final LongAdder episodes = new LongAdder();

	private final AtomicInteger currentScore = new AtomicInteger(0);
	private final AtomicInteger highScore = new AtomicInteger(0);

	private volatile double stepsPerSecond = 0.0;
	private volatile double episodesPerSecond = 0.0;

	private final Thread sampler;
	private volatile boolean running = true;

	public TrainingStats() {
		sampler = new Thread(this::sampleLoop, "stats-sampler");
		sampler.setDaemon(true);
		sampler.start();
	}

	public void onStep(int score) {
		steps.increment();
		currentScore.set(score);
		int hs = highScore.get();
		if (score > hs) highScore.compareAndSet(hs, score);
	}

	public void onEpisodeEnd(int finalScore) {
		episodes.increment();
		currentScore.set(0);
		int hs = highScore.get();
		if (finalScore > hs) highScore.compareAndSet(hs, finalScore);
	}

	private void sampleLoop() {
		long lastSteps = 0L;
		long lastEpisodes = 0L;
		long lastT = System.nanoTime();

		while (running) {
			try {
				Thread.sleep(1000L);
			} catch (InterruptedException e) {
				return;
			}

			long nowT = System.nanoTime();
			double dt = (nowT - lastT) / 1_000_000_000.0;
			if (dt <= 0) dt = 1.0;

			long s = steps.sum();
			long e = episodes.sum();

			stepsPerSecond = (s - lastSteps) / dt;
			episodesPerSecond = (e - lastEpisodes) / dt;

			lastSteps = s;
			lastEpisodes = e;
			lastT = nowT;
		}
	}

	public long getSteps() { return steps.sum(); }
	public long getEpisodes() { return episodes.sum(); }
	public int getCurrentScore() { return currentScore.get(); }
	public int getHighScore() { return highScore.get(); }
	public double getStepsPerSecond() { return stepsPerSecond; }
	public double getEpisodesPerSecond() { return episodesPerSecond; }

	@Override
	public void close() {
		running = false;
		sampler.interrupt();
	}
}
