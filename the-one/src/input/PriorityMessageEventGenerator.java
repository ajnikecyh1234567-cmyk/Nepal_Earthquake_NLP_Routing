/*
 * CSV-driven priority message event generator for The ONE.
 *
 * Reads a fixed simulation schedule with these columns:
 *
 * message_id,creation_time,source,destination,size,
 * true_priority,predicted_priority,tweet_id
 *
 * Supported modes:
 * none       - do not set the routing "priority" property
 * oracle     - priority = true_priority
 * predicted  - priority = predicted_priority
 */

package input;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import core.Settings;
import core.SettingsError;

/**
 * Creates message events from a deterministic CSV schedule.
 */
public class PriorityMessageEventGenerator implements EventQueue {

	/** CSV file setting key. */
	public static final String FILE_S = "file";

	/** Priority selection mode setting key. */
	public static final String MODE_S = "mode";

	/** Supported mode names. */
	public static final String MODE_NONE = "none";
	public static final String MODE_ORACLE = "oracle";
	public static final String MODE_PREDICTED = "predicted";

	private final List<PriorityMessageRecord> records;
	private final String mode;

	private int nextIndex;
	private double nextEventsTime;

	/**
	 * Constructs a CSV-driven event generator.
	 *
	 * Example settings:
	 *
	 * Events1.class = PriorityMessageEventGenerator
	 * Events1.file = data/nepal_simulation_messages.csv
	 * Events1.mode = predicted
	 *
	 * @param settings generator settings
	 */
	public PriorityMessageEventGenerator(Settings settings) {

		String fileName = settings.getSetting(FILE_S);
		this.mode = normaliseMode(
				settings.getSetting(MODE_S));

		this.records = loadRecords(fileName);

		if (this.records.isEmpty()) {
			throw new SettingsError(
					"Priority message CSV contains no records: " +
					fileName);
		}

		Collections.sort(
				this.records,
				new Comparator<PriorityMessageRecord>() {
					@Override
					public int compare(
							PriorityMessageRecord first,
							PriorityMessageRecord second) {

						int timeComparison =
								Double.compare(
										first.creationTime,
										second.creationTime);

						if (timeComparison != 0) {
							return timeComparison;
						}

						return Integer.compare(
								first.lineNumber,
								second.lineNumber);
					}
				});

		this.nextIndex = 0;
		this.nextEventsTime =
				this.records.get(0).creationTime;
	}

	/**
	 * Returns the next CSV-defined message creation event.
	 */
	@Override
	public ExternalEvent nextEvent() {

		if (this.nextIndex >= this.records.size()) {
			this.nextEventsTime = Double.MAX_VALUE;
			return null;
		}

		PriorityMessageRecord record =
				this.records.get(this.nextIndex);

		this.nextIndex++;

		if (this.nextIndex < this.records.size()) {
			this.nextEventsTime =
					this.records.get(
							this.nextIndex).creationTime;
		}
		else {
			this.nextEventsTime = Double.MAX_VALUE;
		}

		return new PriorityMessageCreateEvent(
				record.source,
				record.destination,
				record.messageId,
				record.size,
				record.creationTime,
				record.truePriority,
				record.predictedPriority,
				record.tweetId,
				this.mode);
	}

	/**
	 * Returns the simulation time of the next event.
	 */
	@Override
	public double nextEventsTime() {
		return this.nextEventsTime;
	}

	/**
	 * Loads and validates all rows from the CSV schedule.
	 */
	private List<PriorityMessageRecord> loadRecords(
			String fileName) {

		File file = new File(fileName);

		if (!file.exists()) {
			throw new SettingsError(
					"Priority message CSV was not found: " +
					file.getAbsolutePath());
		}

		List<PriorityMessageRecord> loadedRecords =
				new ArrayList<PriorityMessageRecord>();

		try (BufferedReader reader =
				new BufferedReader(
						new InputStreamReader(
								new FileInputStream(file),
								StandardCharsets.UTF_8))) {

			String headerLine = reader.readLine();

			if (headerLine == null) {
				throw new SettingsError(
						"Priority message CSV is empty: " +
						file.getAbsolutePath());
			}

			headerLine = removeBom(headerLine);

			String[] headerFields =
					splitCsvLine(headerLine);

			Map<String, Integer> columns =
					createColumnMap(headerFields);

			int messageIdColumn =
					requiredColumn(columns, "message_id");
			int creationTimeColumn =
					requiredColumn(columns, "creation_time");
			int sourceColumn =
					requiredColumn(columns, "source");
			int destinationColumn =
					requiredColumn(columns, "destination");
			int sizeColumn =
					requiredColumn(columns, "size");
			int truePriorityColumn =
					requiredColumn(columns, "true_priority");
			int predictedPriorityColumn =
					requiredColumn(
							columns,
							"predicted_priority");
			int tweetIdColumn =
					requiredColumn(columns, "tweet_id");

			int maximumColumn = maximum(
					messageIdColumn,
					creationTimeColumn,
					sourceColumn,
					destinationColumn,
					sizeColumn,
					truePriorityColumn,
					predictedPriorityColumn,
					tweetIdColumn);

			String line;
			int lineNumber = 1;

			while ((line = reader.readLine()) != null) {

				lineNumber++;

				if (line.trim().isEmpty()) {
					continue;
				}

				String[] fields = splitCsvLine(line);

				if (fields.length <= maximumColumn) {
					throw new SettingsError(
							"Not enough columns in " +
							file.getName() +
							" at line " + lineNumber);
				}

				String messageId =
						fields[messageIdColumn].trim();

				double creationTime =
						parseDouble(
								fields[creationTimeColumn],
								"creation_time",
								lineNumber);

				int source =
						parseInteger(
								fields[sourceColumn],
								"source",
								lineNumber);

				int destination =
						parseInteger(
								fields[destinationColumn],
								"destination",
								lineNumber);

				int size =
						parseInteger(
								fields[sizeColumn],
								"size",
								lineNumber);

				int truePriority =
						parsePriority(
								fields[truePriorityColumn],
								"true_priority",
								lineNumber);

				int predictedPriority =
						parsePriority(
								fields[predictedPriorityColumn],
								"predicted_priority",
								lineNumber);

				String tweetId =
						fields[tweetIdColumn].trim();

				validateRecord(
						messageId,
						creationTime,
						source,
						destination,
						size,
						lineNumber);

				loadedRecords.add(
						new PriorityMessageRecord(
								messageId,
								creationTime,
								source,
								destination,
								size,
								truePriority,
								predictedPriority,
								tweetId,
								lineNumber));
			}
		}
		catch (IOException error) {
			throw new SettingsError(
					"Could not read priority message CSV: " +
					file.getAbsolutePath() +
					". Cause: " +
					error.getMessage());
		}

		return loadedRecords;
	}

	/**
	 * Validates a parsed row.
	 */
	private void validateRecord(
			String messageId,
			double creationTime,
			int source,
			int destination,
			int size,
			int lineNumber) {

		if (messageId.isEmpty()) {
			throw new SettingsError(
					"Empty message_id at line " +
					lineNumber);
		}

		if (creationTime < 0) {
			throw new SettingsError(
					"Negative creation_time at line " +
					lineNumber);
		}

		if (source < 0 || destination < 0) {
			throw new SettingsError(
					"Negative host address at line " +
					lineNumber);
		}

		if (source == destination) {
			throw new SettingsError(
					"Source and destination are equal at line " +
					lineNumber);
		}

		if (size <= 0) {
			throw new SettingsError(
					"Message size must be positive at line " +
					lineNumber);
		}
	}

	/**
	 * Converts and validates the configured mode.
	 */
	private String normaliseMode(String configuredMode) {

		String normalised =
				configuredMode.trim().toLowerCase();

		if (!MODE_NONE.equals(normalised) &&
				!MODE_ORACLE.equals(normalised) &&
				!MODE_PREDICTED.equals(normalised)) {

			throw new SettingsError(
					"Invalid priority event mode: " +
					configuredMode +
					". Use none, oracle, or predicted.");
		}

		return normalised;
	}

	/**
	 * Creates a case-insensitive header-name to index map.
	 */
	private Map<String, Integer> createColumnMap(
			String[] headerFields) {

		Map<String, Integer> columns =
				new HashMap<String, Integer>();

		for (int index = 0;
				index < headerFields.length;
				index++) {

			String columnName =
					headerFields[index]
					.trim()
					.toLowerCase();

			columns.put(columnName, index);
		}

		return columns;
	}

	/**
	 * Returns a required header index.
	 */
	private int requiredColumn(
			Map<String, Integer> columns,
			String columnName) {

		Integer index = columns.get(columnName);

		if (index == null) {
			throw new SettingsError(
					"Required CSV column is missing: " +
					columnName);
		}

		return index.intValue();
	}

	/**
	 * Parses an integer field.
	 */
	private int parseInteger(
			String value,
			String columnName,
			int lineNumber) {

		try {
			return Integer.parseInt(value.trim());
		}
		catch (NumberFormatException error) {
			throw new SettingsError(
					"Invalid integer in column " +
					columnName +
					" at line " +
					lineNumber +
					": " +
					value);
		}
	}

	/**
	 * Parses a double field.
	 */
	private double parseDouble(
			String value,
			String columnName,
			int lineNumber) {

		try {
			return Double.parseDouble(value.trim());
		}
		catch (NumberFormatException error) {
			throw new SettingsError(
					"Invalid number in column " +
					columnName +
					" at line " +
					lineNumber +
					": " +
					value);
		}
	}

	/**
	 * Parses a P1-P4 numeric priority.
	 */
	private int parsePriority(
			String value,
			String columnName,
			int lineNumber) {

		int priority =
				parseInteger(
						value,
						columnName,
						lineNumber);

		if (priority < 1 || priority > 4) {
			throw new SettingsError(
					columnName +
					" must be between 1 and 4 at line " +
					lineNumber);
		}

		return priority;
	}

	/**
	 * Splits one CSV line.
	 *
	 * The generated simulation CSV contains only IDs and numeric
	 * fields, so no free-text message column is required here.
	 */
	private String[] splitCsvLine(String line) {
		return line.split(",", -1);
	}

	/**
	 * Removes an optional UTF-8 BOM from the first header field.
	 */
	private String removeBom(String value) {

		if (value.startsWith("\uFEFF")) {
			return value.substring(1);
		}

		return value;
	}

	/**
	 * Returns the largest integer in the supplied values.
	 */
	private int maximum(int... values) {

		int result = Integer.MIN_VALUE;

		for (int value : values) {
			if (value > result) {
				result = value;
			}
		}

		return result;
	}

	/**
	 * One validated row from the simulation CSV.
	 */
	private static class PriorityMessageRecord {

		private final String messageId;
		private final double creationTime;
		private final int source;
		private final int destination;
		private final int size;
		private final int truePriority;
		private final int predictedPriority;
		private final String tweetId;
		private final int lineNumber;

		private PriorityMessageRecord(
				String messageId,
				double creationTime,
				int source,
				int destination,
				int size,
				int truePriority,
				int predictedPriority,
				String tweetId,
				int lineNumber) {

			this.messageId = messageId;
			this.creationTime = creationTime;
			this.source = source;
			this.destination = destination;
			this.size = size;
			this.truePriority = truePriority;
			this.predictedPriority = predictedPriority;
			this.tweetId = tweetId;
			this.lineNumber = lineNumber;
		}
	}
}
