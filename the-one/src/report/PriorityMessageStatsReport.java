/*
 * Priority-specific message statistics report
 *
 * Reports created, delivered, delivery probability, latency,
 * hop count, relay overhead, drops and aborted transfers
 * separately for P1, P2, P3 and P4 messages.
 */

package report;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import core.DTNHost;
import core.Message;
import core.MessageListener;

/**
 * Collects routing statistics separately for priority levels P1-P4.
 *
 * Priority encoding:
 * P1 = 1, highest priority
 * P2 = 2
 * P3 = 3
 * P4 = 4, lowest priority
 */
public class PriorityMessageStatsReport extends Report
		implements MessageListener {

	/** Must match the property used by PrioritySprayAndWaitRouter */
	public static final String PRIORITY_PROPERTY = "priority";

	private static final int MIN_PRIORITY = 1;
	private static final int MAX_PRIORITY = 4;
	private static final int ARRAY_SIZE = 5;

	/** Message creation times, indexed by message ID */
	private Map<String, Double> creationTimes;

	/** Priority recorded when the original message was created */
	private Map<String, Integer> messagePriorities;

	/** Per-priority counters; index 1-4 is used */
	private int[] created;
	private int[] started;
	private int[] relayed;
	private int[] delivered;
	private int[] aborted;
	private int[] dropped;
	private int[] removed;

	/** Per-priority delivery latency values */
	private Map<Integer, List<Double>> latencies;

	/** Per-priority delivered-message hop counts */
	private Map<Integer, List<Integer>> hopCounts;

	/** Number of created messages without a valid priority property */
	private int messagesWithoutPriority;

	/**
	 * Constructor.
	 */
	public PriorityMessageStatsReport() {
		init();
	}

	/**
	 * Initializes all report data structures.
	 */
	@Override
	protected void init() {
		super.init();

		this.creationTimes = new HashMap<String, Double>();
		this.messagePriorities = new HashMap<String, Integer>();

		this.created = new int[ARRAY_SIZE];
		this.started = new int[ARRAY_SIZE];
		this.relayed = new int[ARRAY_SIZE];
		this.delivered = new int[ARRAY_SIZE];
		this.aborted = new int[ARRAY_SIZE];
		this.dropped = new int[ARRAY_SIZE];
		this.removed = new int[ARRAY_SIZE];

		this.latencies =
				new HashMap<Integer, List<Double>>();

		this.hopCounts =
				new HashMap<Integer, List<Integer>>();

		for (int priority = MIN_PRIORITY;
				priority <= MAX_PRIORITY;
				priority++) {

			Integer key = Integer.valueOf(priority);

			this.latencies.put(
					key,
					new ArrayList<Double>());

			this.hopCounts.put(
					key,
					new ArrayList<Integer>());
		}

		this.messagesWithoutPriority = 0;
	}

	/**
	 * Called when a new original message is created.
	 */
	@Override
	public void newMessage(Message message) {

		if (isWarmup()) {
			addWarmupID(message.getId());
			return;
		}

		int priority = readPriority(message);

		/*
		 * Detect missing or invalid priorities.
		 * The fallback priority is P4.
		 */
		if (!hasValidPriority(message)) {
			this.messagesWithoutPriority++;
		}

		this.creationTimes.put(
				message.getId(),
				Double.valueOf(getSimTime()));

		this.messagePriorities.put(
				message.getId(),
				Integer.valueOf(priority));

		this.created[priority]++;
	}

	/**
	 * Called when a transfer attempt starts.
	 */
	@Override
	public void messageTransferStarted(
			Message message,
			DTNHost from,
			DTNHost to) {

		if (isWarmupID(message.getId())) {
			return;
		}

		int priority = getStoredOrMessagePriority(message);

		this.started[priority]++;
	}

	/**
	 * Called when a transfer is successfully completed.
	 */
	@Override
	public void messageTransferred(
			Message message,
			DTNHost from,
			DTNHost to,
			boolean finalTarget) {

		if (isWarmupID(message.getId())) {
			return;
		}

		int priority = getStoredOrMessagePriority(message);

		this.relayed[priority]++;

		/*
		 * finalTarget is true only for the first successful delivery
		 * to the final destination.
		 */
		if (finalTarget) {

			this.delivered[priority]++;

			Double creationTime =
					this.creationTimes.get(message.getId());

			if (creationTime != null) {
				double latency =
						getSimTime() -
						creationTime.doubleValue();

				this.latencies
						.get(Integer.valueOf(priority))
						.add(Double.valueOf(latency));
			}

			int hops = message.getHops().size() - 1;

			this.hopCounts
					.get(Integer.valueOf(priority))
					.add(Integer.valueOf(hops));
		}
	}

	/**
	 * Called when an ongoing transfer is aborted.
	 */
	@Override
	public void messageTransferAborted(
			Message message,
			DTNHost from,
			DTNHost to) {

		if (isWarmupID(message.getId())) {
			return;
		}

		int priority = getStoredOrMessagePriority(message);

		this.aborted[priority]++;
	}

	/**
	 * Called when a message copy is deleted from a node buffer.
	 */
	@Override
	public void messageDeleted(
			Message message,
			DTNHost where,
			boolean wasDropped) {

		if (isWarmupID(message.getId())) {
			return;
		}

		int priority = getStoredOrMessagePriority(message);

		if (wasDropped) {
			this.dropped[priority]++;
		}
		else {
			this.removed[priority]++;
		}
	}

	/**
	 * Returns true when the message contains a valid P1-P4 property.
	 */
	private boolean hasValidPriority(Message message) {

		Object value =
				message.getProperty(PRIORITY_PROPERTY);

		if (!(value instanceof Number)) {
			return false;
		}

		int priority =
				((Number) value).intValue();

		return priority >= MIN_PRIORITY &&
				priority <= MAX_PRIORITY;
	}

	/**
	 * Reads the priority property.
	 *
	 * Missing or invalid priorities are treated as P4.
	 */
	private int readPriority(Message message) {

		Object value =
				message.getProperty(PRIORITY_PROPERTY);

		if (value instanceof Number) {

			int priority =
					((Number) value).intValue();

			if (priority >= MIN_PRIORITY &&
					priority <= MAX_PRIORITY) {

				return priority;
			}
		}

		return MAX_PRIORITY;
	}

	/**
	 * Uses the priority stored when the original message was created.
	 * If no stored value exists, reads the property from the current copy.
	 */
	private int getStoredOrMessagePriority(Message message) {

		Integer storedPriority =
				this.messagePriorities.get(message.getId());

		if (storedPriority != null) {
			return storedPriority.intValue();
		}

		int priority = readPriority(message);

		this.messagePriorities.put(
				message.getId(),
				Integer.valueOf(priority));

		return priority;
	}

	/**
	 * Writes the final P1-P4 statistics.
	 */
	@Override
	public void done() {

		write(
			"Priority message statistics for scenario " +
			getScenarioName() +
			"\nsim_time: " +
			format(getSimTime()) +
			"\nmessages_without_priority: " +
			this.messagesWithoutPriority +
			"\n");

		write(
			"priority created delivered delivery_prob " +
			"started relayed overhead_ratio aborted " +
			"dropped removed latency_avg latency_med " +
			"hopcount_avg hopcount_med");

		for (int priority = MIN_PRIORITY;
				priority <= MAX_PRIORITY;
				priority++) {

			double deliveryProbability = 0.0;
			double overheadRatio = Double.NaN;

			if (this.created[priority] > 0) {
				deliveryProbability =
					(1.0 * this.delivered[priority]) /
					this.created[priority];
			}

			if (this.delivered[priority] > 0) {
				overheadRatio =
					(1.0 *
						(this.relayed[priority] -
						 this.delivered[priority])) /
					this.delivered[priority];
			}

			List<Double> priorityLatencies =
					this.latencies.get(
							Integer.valueOf(priority));

			List<Integer> priorityHopCounts =
					this.hopCounts.get(
							Integer.valueOf(priority));

			String line =
					"P" + priority +
					" " + this.created[priority] +
					" " + this.delivered[priority] +
					" " + format(deliveryProbability) +
					" " + this.started[priority] +
					" " + this.relayed[priority] +
					" " + format(overheadRatio) +
					" " + this.aborted[priority] +
					" " + this.dropped[priority] +
					" " + this.removed[priority] +
					" " + getAverage(priorityLatencies) +
					" " + getMedian(priorityLatencies) +
					" " + getIntAverage(priorityHopCounts) +
					" " + getIntMedian(priorityHopCounts);

			write(line);
		}

		super.done();
	}
}