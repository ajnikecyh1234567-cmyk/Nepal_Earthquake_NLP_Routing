/*
 * Priority-aware message transfer and buffer event log.
 *
 * Intended for short debugging and controlled verification tests.
 *
 * Recorded events:
 * STARTED   - a message transfer attempt started
 * RELAYED   - a message was transferred to an intermediate node
 * DELIVERED - a message reached its final destination
 * ABORTED   - a transfer was interrupted
 * DROPPED   - a message copy was deleted because of buffer/TTL pressure
 * REMOVED   - a message copy was removed normally
 */

package report;

import core.DTNHost;
import core.Message;
import core.MessageListener;

public class PriorityTransferLogReport extends Report
		implements MessageListener {


	public static final String PRIORITY_PROPERTY = "priority";


	public static final String COPY_PROPERTY =
			"SprayAndWaitRouter.copies";


	public PriorityTransferLogReport() {
		init();
	}


	@Override
	protected void init() {

		super.init();

		write(
			"time sender receiver message " +
			"priority copies event"
		);
	}


	@Override
	public void newMessage(Message message) {

		if (isWarmup()) {
			addWarmupID(message.getId());
		}
	}


	@Override
	public void messageTransferStarted(
			Message message,
			DTNHost from,
			DTNHost to) {

		if (isWarmupID(message.getId())) {
			return;
		}

		writeTransferEvent(
				message,
				from,
				to,
				"STARTED");
	}


	@Override
	public void messageTransferred(
			Message message,
			DTNHost from,
			DTNHost to,
			boolean finalTarget) {

		if (isWarmupID(message.getId())) {
			return;
		}

		String event;

		if (finalTarget) {
			event = "DELIVERED";
		}
		else {
			event = "RELAYED";
		}

		writeTransferEvent(
				message,
				from,
				to,
				event);
	}


	@Override
	public void messageTransferAborted(
			Message message,
			DTNHost from,
			DTNHost to) {

		if (isWarmupID(message.getId())) {
			return;
		}

		writeTransferEvent(
				message,
				from,
				to,
				"ABORTED");
	}

	
	@Override
	public void messageDeleted(
			Message message,
			DTNHost where,
			boolean wasDropped) {

		if (isWarmupID(message.getId())) {
			return;
		}

		String event;

		if (wasDropped) {
			event = "DROPPED";
		}
		else {
			event = "REMOVED";
		}

		int priority = readPriority(message);
		int copies = readCopies(message);

		write(
			format(getSimTime()) + " " +
			where + " " +
			"-" + " " +
			message.getId() + " " +
			"P" + priority + " " +
			copies + " " +
			event
		);
	}


	private void writeTransferEvent(
			Message message,
			DTNHost from,
			DTNHost to,
			String event) {

		int priority = readPriority(message);
		int copies = readCopies(message);

		write(
			format(getSimTime()) + " " +
			from + " " +
			to + " " +
			message.getId() + " " +
			"P" + priority + " " +
			copies + " " +
			event
		);
	}


	private int readPriority(Message message) {

		Object value =
				message.getProperty(
						PRIORITY_PROPERTY);

		if (value instanceof Number) {

			int priority =
					((Number) value).intValue();

			if (priority >= 1 &&
					priority <= 4) {

				return priority;
			}
		}

		return 4;
	}


	private int readCopies(Message message) {

		Object value =
				message.getProperty(
						COPY_PROPERTY);

		if (value instanceof Number) {

			return ((Number) value).intValue();
		}

		return -1;
	}

	@Override
	public void done() {

		super.done();
	}
}