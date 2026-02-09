package dev.matito.snake_ai.dqn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class ReplayBuffer {
	private final int capacity;
	private final List<Transition> buffer;
	private final Random random;

	public ReplayBuffer(int capacity, long seed) {
		this.capacity = capacity;
		this.buffer = new ArrayList<>(capacity);
		this.random = new Random(seed);
	}

	public void add(Transition transition) {
		if (buffer.size() >= capacity) {
			buffer.removeFirst();
		}
		buffer.add(transition);
	}

	public List<Transition> sample(int batchSize) {
		if (buffer.size() < batchSize) {
			return new ArrayList<>(buffer);
		}

		List<Transition> batch = new ArrayList<>(batchSize);
		List<Integer> indices = new ArrayList<>(buffer.size());
		for (int i = 0; i < buffer.size(); i++) {
			indices.add(i);
		}

		for (int i = 0; i < batchSize; i++) {
			int idx = random.nextInt(indices.size());
			batch.add(buffer.get(indices.remove(idx)));
		}

		return batch;
	}

	public int size() {
		return buffer.size();
	}

	public void clear() {
		buffer.clear();
	}
}