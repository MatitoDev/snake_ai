package dev.matito.snake_ai.dashboard;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import dev.matito.snake_ai.dqn.DQNAgent;
import dev.matito.snake_ai.ql.QLAgent;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

public final class DashboardServer implements AutoCloseable {

	private final HttpServer server;
	private final ScheduledExecutorService scheduler;
	private final AtomicReference<String> statusJson = new AtomicReference<>("{}");
	private final AtomicReference<String> nnSummaryJson = new AtomicReference<>("{}");

	private final String configPath;
	private final Object agent;
	private final TrainingStats stats;

	private volatile Map<String, String> configCache = Map.of();
	private volatile long configLastModified = -1L;

	private DashboardServer(HttpServer server, String configPath, Object agent, TrainingStats stats) {
		this.server = server;
		this.configPath = configPath;
		this.agent = agent;
		this.stats = stats;

		this.scheduler = Executors.newSingleThreadScheduledExecutor(r -> {
			Thread t = new Thread(r, "dashboard-snapshot");
			t.setDaemon(true);
			return t;
		});
	}

	public static DashboardServer start(String configPath, Object agent, TrainingStats stats) {
		Properties p = loadPropsQuiet(configPath);

		boolean enabled = Boolean.parseBoolean(p.getProperty("dashboard.enabled", "true").trim());
		if (!enabled) return null;

		String bind = p.getProperty("dashboard.bind", "127.0.0.1").trim();
		int port = parseInt(p.getProperty("dashboard.port", "8080"), 8080);

		HttpServer http;
		try {
			http = HttpServer.create(new InetSocketAddress(bind, port), 0);
		} catch (IOException e) {
			throw new RuntimeException("Dashboard bind failed: " + bind + ":" + port, e);
		}

		DashboardServer ds = new DashboardServer(http, configPath, agent, stats);
		ds.installHandlers();
		ds.startSnapshotLoop();

		http.setExecutor(Executors.newCachedThreadPool(r -> {
			Thread t = new Thread(r, "dashboard-http");
			t.setDaemon(true);
			return t;
		}));
		http.start();

		System.out.println("Dashboard: " + bind + ":" + port);
		return ds;
	}

	private void installHandlers() {
		server.createContext("/", ex -> {
			if (!isGet(ex)) return;
			sendText(ex, 200, "text/html; charset=utf-8", DashboardHtml.PAGE);
		});

		server.createContext("/api/status", ex -> {
			if (!isGet(ex)) return;
			sendText(ex, 200, "application/json; charset=utf-8", statusJson.get());
		});

		server.createContext("/api/qtable", ex -> {
			if (!isGet(ex)) return;
			if (!(agent instanceof QLAgent ql)) {
				sendText(ex, 404, "application/json; charset=utf-8", "{\"error\":\"not_ql\"}");
				return;
			}

			Map<String, String> q = query(ex.getRequestURI());
			int start = parseInt(q.getOrDefault("start", "0"), 0);
			int limit = parseInt(q.getOrDefault("limit", "128"), 128);

			limit = Math.max(1, Math.min(2048, limit));
			start = Math.max(0, Math.min(QLAgent.getNumStates() - 1, start));
			int end = Math.min(QLAgent.getNumStates(), start + limit);

			double[][] slice = ql.getQSlice(start, end - start);

			StringBuilder sb = new StringBuilder(64 * (end - start));
			sb.append("{\"start\":").append(start)
					.append(",\"limit\":").append(end - start)
					.append(",\"states\":").append(QLAgent.getNumStates())
					.append(",\"actions\":").append(QLAgent.getNumActions())
					.append(",\"rows\":[");

			for (int i = 0; i < slice.length; i++) {
				if (i > 0) sb.append(',');
				sb.append('[')
						.append(formatDouble(slice[i][0])).append(',')
						.append(formatDouble(slice[i][1])).append(',')
						.append(formatDouble(slice[i][2]))
						.append(']');
			}
			sb.append("]}");

			sendText(ex, 200, "application/json; charset=utf-8", sb.toString());
		});

		server.createContext("/api/nn/summary", ex -> {
			if (!isGet(ex)) return;
			if (!(agent instanceof DQNAgent)) {
				sendText(ex, 404, "application/json; charset=utf-8", "{\"error\":\"not_dqn\"}");
				return;
			}
			sendText(ex, 200, "application/json; charset=utf-8", nnSummaryJson.get());
		});

		server.createContext("/api/nn/slice", ex -> {
			if (!isGet(ex)) return;
			if (!(agent instanceof DQNAgent dqn)) {
				sendText(ex, 404, "application/json; charset=utf-8", "{\"error\":\"not_dqn\"}");
				return;
			}

			Map<String, String> q = query(ex.getRequestURI());
			String net = q.getOrDefault("net", "online");

			int inStart = parseInt(q.getOrDefault("inStart", "0"), 0);
			int inCount = parseInt(q.getOrDefault("inCount", "6"), 6);

			int h1Start = parseInt(q.getOrDefault("h1Start", "0"), 0);
			int h1Count = parseInt(q.getOrDefault("h1Count", "8"), 8);

			int h2Start = parseInt(q.getOrDefault("h2Start", "0"), 0);
			int h2Count = parseInt(q.getOrDefault("h2Count", "8"), 8);

			String json = dqn.getNetworkSliceJson(net, inStart, inCount, h1Start, h1Count, h2Start, h2Count);
			sendText(ex, 200, "application/json; charset=utf-8", json);
		});

	}

	private void startSnapshotLoop() {
		long refreshMs = parseLong(loadPropsQuiet(configPath).getProperty("dashboard.refresh.ms", "500"), 500L);
		refreshMs = Math.max(200L, Math.min(5000L, refreshMs));

		scheduler.scheduleAtFixedRate(() -> {
			try {
				refreshConfigIfNeeded();
				buildStatusSnapshot();
				buildNnSummarySnapshot();
			} catch (Throwable ignored) {
			}
		}, 0L, refreshMs, TimeUnit.MILLISECONDS);
	}

	private void refreshConfigIfNeeded() {
		try {
			Path path = Path.of(configPath);
			long lm = Files.exists(path) ? Files.getLastModifiedTime(path).toMillis() : -1L;
			if (lm == configLastModified) return;

			Properties p = loadPropsQuiet(configPath);
			Map<String, String> cfg = new LinkedHashMap<>();
			for (String k : p.stringPropertyNames()) cfg.put(k, p.getProperty(k));
			configCache = Map.copyOf(cfg);
			configLastModified = lm;
		} catch (IOException ignored) {
		}
	}

	private void buildStatusSnapshot() {
		StringBuilder sb = new StringBuilder(2048);
		sb.append('{');

		sb.append("\"config\":");
		appendStringMap(sb, configCache);

		sb.append(",\"metrics\":{");
		sb.append("\"highscore\":").append(stats.getHighScore()).append(',');
		sb.append("\"currentScore\":").append(stats.getCurrentScore()).append(',');
		sb.append("\"episodes\":").append(stats.getEpisodes()).append(',');
		sb.append("\"steps\":").append(stats.getSteps()).append(',');
		sb.append("\"episodesPerSec\":").append(formatDouble(stats.getEpisodesPerSecond())).append(',');
		sb.append("\"stepsPerSec\":").append(formatDouble(stats.getStepsPerSecond()));
		sb.append('}');

		if (agent instanceof QLAgent ql) {
			int s = ql.getLastStateId();
			sb.append(",\"agent\":{");
			sb.append("\"type\":\"QL\",");
			sb.append("\"alpha\":").append(formatDouble(ql.getAlpha())).append(',');
			sb.append("\"gamma\":").append(formatDouble(ql.getGamma())).append(',');
			sb.append("\"epsilon\":").append(formatDouble(ql.getEpsilon())).append(',');
			sb.append("\"epsilonMin\":").append(formatDouble(ql.getEpsilonMin())).append(',');
			sb.append("\"lastStateId\":").append(s).append(',');
			sb.append("\"lastAction\":").append(ql.getLastAction()).append(',');
			sb.append("\"qRow\":");
			appendDoubleArray3(sb, (s >= 0) ? ql.getQRow(s) : null);
			sb.append('}');
		} else if (agent instanceof DQNAgent dqn) {
			sb.append(",\"agent\":{");
			sb.append("\"type\":\"DQN\",");
			sb.append("\"alpha\":").append(formatDouble(dqn.getLearningRate())).append(',');
			sb.append("\"gamma\":").append(formatDouble(dqn.getGamma())).append(',');
			sb.append("\"epsilon\":").append(formatDouble(dqn.getEpsilon())).append(',');
			sb.append("\"epsilonMin\":").append(formatDouble(dqn.getEpsilonMin())).append(',');
			sb.append("\"replaySize\":").append(dqn.getReplayBufferSize()).append(',');
			sb.append("\"lastQ\":");
			appendDoubleArray3(sb, dqn.getLastQValues());
			sb.append('}');
		} else {
			sb.append(",\"agent\":{");
			sb.append("\"type\":\"unknown\"}");
		}

		sb.append('}');
		statusJson.set(sb.toString());
	}

	private void buildNnSummarySnapshot() {
		if (!(agent instanceof DQNAgent dqn)) return;
		String s = dqn.getOnlineNetworkSummaryJson();
		if (s != null && !s.isEmpty()) nnSummaryJson.set(s);
	}

	private static void appendStringMap(StringBuilder sb, Map<String, String> map) {
		sb.append('{');
		int i = 0;
		for (Map.Entry<String, String> e : map.entrySet()) {
			if (i++ > 0) sb.append(',');
			sb.append('\"').append(jsonEscape(e.getKey())).append("\":");
			sb.append('\"').append(jsonEscape(e.getValue())).append('\"');
		}
		sb.append('}');
	}

	private static void appendDoubleArray3(StringBuilder sb, double[] arr) {
		sb.append('[');
		if (arr != null && arr.length >= 3) {
			sb.append(formatDouble(arr[0])).append(',')
					.append(formatDouble(arr[1])).append(',')
					.append(formatDouble(arr[2]));
		} else {
			sb.append("null,null,null");
		}
		sb.append(']');
	}

	private static boolean isGet(HttpExchange ex) throws IOException {
		if ("GET".equalsIgnoreCase(ex.getRequestMethod())) return true;
		ex.getResponseHeaders().set("Allow", "GET");
		sendText(ex, 405, "text/plain; charset=utf-8", "Method Not Allowed");
		return false;
	}

	private static void sendText(HttpExchange ex, int status, String contentType, String body) throws IOException {
		byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
		Headers h = ex.getResponseHeaders();
		h.set("Content-Type", contentType);
		h.set("Cache-Control", "no-store");
		ex.sendResponseHeaders(status, bytes.length);
		try (OutputStream os = ex.getResponseBody()) {
			os.write(bytes);
		}
	}

	private static Map<String, String> query(URI uri) {
		String q = uri.getRawQuery();
		if (q == null || q.isEmpty()) return Map.of();
		Map<String, String> out = new LinkedHashMap<>();
		String[] parts = q.split("&");
		for (String part : parts) {
			int eq = part.indexOf('=');
			if (eq <= 0) continue;
			String k = urlDecode(part.substring(0, eq));
			String v = urlDecode(part.substring(eq + 1));
			out.put(k, v);
		}
		return out;
	}

	private static String urlDecode(String s) {
		return s.replace("+", " ");
	}

	private static Properties loadPropsQuiet(String configPath) {
		Properties p = new Properties();
		try {
			Path path = Path.of(configPath);
			if (Files.exists(path)) {
				try (var in = Files.newInputStream(path)) {
					p.load(in);
				}
			}
		} catch (IOException ignored) {
		}
		return p;
	}

	private static int parseInt(String s, int def) {
		try { return Integer.parseInt(s.trim()); } catch (Exception e) { return def; }
	}

	private static long parseLong(String s, long def) {
		try { return Long.parseLong(s.trim()); } catch (Exception e) { return def; }
	}

	private static String jsonEscape(String s) {
		if (s == null) return "";
		StringBuilder sb = new StringBuilder(s.length() + 16);
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			switch (c) {
				case '"' -> sb.append("\\\"");
				case '\\' -> sb.append("\\\\");
				case '\n' -> sb.append("\\n");
				case '\r' -> sb.append("\\r");
				case '\t' -> sb.append("\\t");
				default -> {
					if (c < 0x20) sb.append(' ');
					else sb.append(c);
				}
			}
		}
		return sb.toString();
	}

	private static String formatDouble(double v) {
		if (Double.isNaN(v) || Double.isInfinite(v)) return "0.0";
		return Double.toString(v);
	}

	@Override
	public void close() {
		try { scheduler.shutdownNow(); } catch (Exception ignored) {}
		try { server.stop(0); } catch (Exception ignored) {}
	}
}
