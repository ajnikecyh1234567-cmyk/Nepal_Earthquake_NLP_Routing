package routing;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import core.Connection;
import core.DTNHost;
import core.Message;
import core.Settings;


public class PrioritySprayAndWaitRouter_V2_backup extends ActiveRouter {

	
	public static final String NROF_COPIES = "nrofCopies";

	
	public static final String BINARY_MODE = "binaryMode";

	
	public static final String SPRAYANDWAIT_NS =
			"SprayAndWaitRouter";


	public static final String MSG_COUNT_PROPERTY =
			SPRAYANDWAIT_NS + ".copies";


	public static final String PRIORITY_PROPERTY =
			"priority";


	public static final int PRIORITY_P1 = 1;
	public static final int PRIORITY_P2 = 2;
	public static final int PRIORITY_P3 = 3;
	public static final int PRIORITY_P4 = 4;


	protected int initialNrofCopies;


	protected boolean isBinary;


	public PrioritySprayAndWaitRouter_V2_backup(Settings settings) {

		super(settings);

		Settings snwSettings =
				new Settings(SPRAYANDWAIT_NS);

		this.initialNrofCopies =
				snwSettings.getInt(NROF_COPIES);

		this.isBinary =
				snwSettings.getBoolean(BINARY_MODE);
	}

	
	protected PrioritySprayAndWaitRouter_V2_backup(
			PrioritySprayAndWaitRouter_V2_backup router) {

		super(router);

		this.initialNrofCopies =
				router.initialNrofCopies;

		this.isBinary =
				router.isBinary;
	}


	@Override
	public Message messageTransferred(
			String id,
			DTNHost from) {

		Message message =
				super.messageTransferred(id, from);

		Integer numberOfCopies =
				(Integer) message.getProperty(
						MSG_COUNT_PROPERTY);

		assert numberOfCopies != null :
				"Not a Spray-and-Wait message: " +
				message;

		if (this.isBinary) {

			numberOfCopies =
					Integer.valueOf(
							(int) Math.floor(
									numberOfCopies.intValue()
											/ 2.0));
		}
		else {

			numberOfCopies =
					Integer.valueOf(1);
		}

		message.updateProperty(
				MSG_COUNT_PROPERTY,
				numberOfCopies);

		return message;
	}

	
	@Override
	public boolean createNewMessage(Message message) {

		makeRoomForNewMessage(message.getSize());

		message.setTtl(this.msgTtl);

		
		if (message.getProperty(PRIORITY_PROPERTY) == null) {

			int priority =
					deriveTestPriority(message);

			message.addProperty(
					PRIORITY_PROPERTY,
					Integer.valueOf(priority));
		}

		message.addProperty(
				MSG_COUNT_PROPERTY,
				Integer.valueOf(
						this.initialNrofCopies));

		addToMessages(message, true);

		return true;
	}

	
	@Override
	public void update() {

		super.update();

		if (!canStartTransfer() ||
				isTransferring()) {

			return;
		}

		
		if (exchangeDeliverableMessages() != null) {
			return;
		}

		List<Message> copiesLeft =
				getMessagesWithCopiesLeft();

		
		@SuppressWarnings(value = "unchecked")
		List<Message> queueSortedMessages =
				sortByQueueMode(copiesLeft);

		
		Collections.sort(
				queueSortedMessages,
				new Comparator<Message>() {

					@Override
					public int compare(
							Message first,
							Message second) {

						int firstPriority =
								getMessagePriority(first);

						int secondPriority =
								getMessagePriority(second);

						if (firstPriority <
								secondPriority) {

							return -1;
						}

						if (firstPriority >
								secondPriority) {

							return 1;
						}

						return 0;
					}
				});

		if (!queueSortedMessages.isEmpty()) {

			tryMessagesToConnections(
					queueSortedMessages,
					getConnections());
		}
	}

	
	protected List<Message> getMessagesWithCopiesLeft() {

		List<Message> messages =
				new ArrayList<Message>();

		for (Message message :
				getMessageCollection()) {

			Integer numberOfCopies =
					(Integer) message.getProperty(
							MSG_COUNT_PROPERTY);

			assert numberOfCopies != null :
					"Spray-and-Wait message " +
					message +
					" does not have a copies property";

			if (numberOfCopies.intValue() > 1) {

				messages.add(message);
			}
		}

		return messages;
	}

	
	protected int getMessagePriority(
			Message message) {

		Object value =
				message.getProperty(
						PRIORITY_PROPERTY);

		if (value instanceof Number) {

			int priority =
					((Number) value).intValue();

			if (priority >= PRIORITY_P1 &&
					priority <= PRIORITY_P4) {

				return priority;
			}
		}

		
		return PRIORITY_P4;
	}

	
	protected int deriveTestPriority(
			Message message) {

		String id = message.getId();

		int numericPart = 0;
		boolean foundDigit = false;

		for (int index = 0;
				index < id.length();
				index++) {

			char character =
					id.charAt(index);

			if (Character.isDigit(character)) {

				foundDigit = true;

				numericPart =
						numericPart * 10 +
						(character - '0');
			}
		}

		if (foundDigit && numericPart > 0) {

			return ((numericPart - 1) % 4) + 1;
		}

		
		int positiveHash =
				id.hashCode() & 0x7fffffff;

		return (positiveHash % 4) + 1;
	}


	@Override
	protected Message getNextMessageToRemove(
			boolean excludeMsgBeingSent) {

		Message candidate = null;

		for (Message message :
				getMessageCollection()) {

			/*
			 * Never select a message that is currently being sent.
			 */
			if (excludeMsgBeingSent &&
					isSending(message.getId())) {

				continue;
			}

			if (candidate == null) {

				candidate = message;
				continue;
			}

			int messagePriority =
					getMessagePriority(message);

			int candidatePriority =
					getMessagePriority(candidate);

			
			if (messagePriority >
					candidatePriority) {

				candidate = message;
			}
			else if (messagePriority ==
					candidatePriority) {

				
				if (message.getReceiveTime() <
						candidate.getReceiveTime()) {

					candidate = message;
				}
			}
		}

		return candidate;
	}

	
	@Override
	protected void transferDone(
			Connection connection) {

		String messageId =
				connection.getMessage().getId();

		Message message =
				getMessage(messageId);

		if (message == null) {

			
			return;
		}

		Integer numberOfCopies =
				(Integer) message.getProperty(
						MSG_COUNT_PROPERTY);

		assert numberOfCopies != null :
				"Message does not have a copies property: " +
				message;

		if (this.isBinary) {

			numberOfCopies =
					Integer.valueOf(
							(int) Math.ceil(
									numberOfCopies.intValue()
											/ 2.0));
		}
		else {

			numberOfCopies =
					Integer.valueOf(
							numberOfCopies.intValue() - 1);
		}

		message.updateProperty(
				MSG_COUNT_PROPERTY,
				numberOfCopies);
	}

	
	@Override
	public PrioritySprayAndWaitRouter_V2_backup replicate() {

		return new PrioritySprayAndWaitRouter_V2_backup(this);
	}
}