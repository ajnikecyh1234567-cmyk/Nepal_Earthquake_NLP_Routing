/*
 * Priority-aware external event for creating one message.
 */

package input;

import core.DTNHost;
import core.Message;
import core.World;

/**
 * Creates a message and writes the NLP priority properties before
 * the source host passes the message to its router.
 */
public class PriorityMessageCreateEvent extends MessageEvent {

	public static final String PRIORITY_PROPERTY = "priority";
	public static final String TRUE_PRIORITY_PROPERTY =
			"truePriority";
	public static final String PREDICTED_PRIORITY_PROPERTY =
			"predictedPriority";
	public static final String TWEET_ID_PROPERTY = "tweetId";

	private final int size;
	private final int truePriority;
	private final int predictedPriority;
	private final String tweetId;
	private final String mode;

	/**
	 * Creates a CSV-defined message creation event.
	 *
	 * @param from source node address
	 * @param to destination node address
	 * @param id message ID
	 * @param size message size in bytes
	 * @param time message creation time
	 * @param truePriority ground-truth P1-P4 priority
	 * @param predictedPriority NLP-predicted P1-P4 priority
	 * @param tweetId original tweet identifier
	 * @param mode none, oracle or predicted
	 */
	public PriorityMessageCreateEvent(
			int from,
			int to,
			String id,
			int size,
			double time,
			int truePriority,
			int predictedPriority,
			String tweetId,
			String mode) {

		super(from, to, id, time);

		this.size = size;
		this.truePriority = truePriority;
		this.predictedPriority = predictedPriority;
		this.tweetId = tweetId;
		this.mode = mode;
	}

	/**
	 * Creates the message and attaches all priority metadata before
	 * handing it to the source host.
	 */
	@Override
	public void processEvent(World world) {

		DTNHost to =
				world.getNodeByAddress(this.toAddr);

		DTNHost from =
				world.getNodeByAddress(this.fromAddr);

		Message message =
				new Message(
						from,
						to,
						this.id,
						this.size);

		/* This project uses one-way emergency messages. */
		message.setResponseSize(0);

		/* Always preserve both the ground truth and prediction. */
		message.addProperty(
				TRUE_PRIORITY_PROPERTY,
				Integer.valueOf(this.truePriority));

		message.addProperty(
				PREDICTED_PRIORITY_PROPERTY,
				Integer.valueOf(this.predictedPriority));

		if (this.tweetId != null &&
				!this.tweetId.isEmpty()) {

			message.addProperty(
					TWEET_ID_PROPERTY,
					this.tweetId);
		}

		/*
		 * Select the priority actually used by the routing policy.
		 *
		 * none:
		 *   no "priority" property is added
		 *
		 * oracle:
		 *   priority = truePriority
		 *
		 * predicted:
		 *   priority = predictedPriority
		 */
		if (PriorityMessageEventGenerator.MODE_ORACLE.equals(
				this.mode)) {

			message.addProperty(
					PRIORITY_PROPERTY,
					Integer.valueOf(this.truePriority));
		}
		else if (
				PriorityMessageEventGenerator.MODE_PREDICTED.equals(
						this.mode)) {

			message.addProperty(
					PRIORITY_PROPERTY,
					Integer.valueOf(this.predictedPriority));
		}
		else if (
				!PriorityMessageEventGenerator.MODE_NONE.equals(
						this.mode)) {

			throw new IllegalStateException(
					"Unsupported priority mode: " +
					this.mode);
		}

		/*
		 * The properties must be written before this call so that
		 * PrioritySprayAndWaitRouter can read "priority" while
		 * creating the message.
		 */
		from.createNewMessage(message);
	}

	@Override
	public String toString() {

		return super.toString() +
				" [" +
				this.fromAddr +
				"->" +
				this.toAddr +
				"] size:" +
				this.size +
				" true:P" +
				this.truePriority +
				" predicted:P" +
				this.predictedPriority +
				" mode:" +
				this.mode +
				" CREATE";
	}
}
