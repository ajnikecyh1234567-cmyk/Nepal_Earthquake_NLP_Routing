/*
 * True-priority message statistics report for The ONE.
 *
 * This report always groups messages by the "truePriority" property,
 * regardless of which priority the router actually used.
 */

package report;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import core.DTNHost;
import core.Message;
import core.MessageListener;

/**
 * Reports message statistics separately for true P1-P4 classes.
 */
public class TruePriorityMessageStatsReport extends Report
        implements MessageListener {

    public static final String TRUE_PRIORITY_PROPERTY = "truePriority";

    private static final int MIN_PRIORITY = 1;
    private static final int MAX_PRIORITY = 4;

    private int[] created;
    private int[] delivered;
    private int[] started;
    private int[] relayed;
    private int[] aborted;
    private int[] dropped;
    private int[] removed;

    private List<Double>[] latencies;
    private List<Integer>[] hopCounts;

    private int messagesWithoutTruePriority;

    public TruePriorityMessageStatsReport() {
        init();
    }

    @Override
    @SuppressWarnings("unchecked")
    protected void init() {
        super.init();

        this.created = new int[MAX_PRIORITY + 1];
        this.delivered = new int[MAX_PRIORITY + 1];
        this.started = new int[MAX_PRIORITY + 1];
        this.relayed = new int[MAX_PRIORITY + 1];
        this.aborted = new int[MAX_PRIORITY + 1];
        this.dropped = new int[MAX_PRIORITY + 1];
        this.removed = new int[MAX_PRIORITY + 1];

        this.latencies = (List<Double>[]) new List<?>[MAX_PRIORITY + 1];
        this.hopCounts = (List<Integer>[]) new List<?>[MAX_PRIORITY + 1];

        for (int priority = MIN_PRIORITY;
                priority <= MAX_PRIORITY;
                priority++) {
            this.latencies[priority] = new ArrayList<Double>();
            this.hopCounts[priority] = new ArrayList<Integer>();
        }

        this.messagesWithoutTruePriority = 0;
    }

    @Override
    public void newMessage(Message message) {
        if (isWarmup()) {
            addWarmupID(message.getId());
            return;
        }

        int priority = readTruePriority(message);

        if (!isValidPriority(priority)) {
            this.messagesWithoutTruePriority++;
            return;
        }

        this.created[priority]++;
    }

    @Override
    public void messageTransferStarted(
            Message message,
            DTNHost from,
            DTNHost to) {

        if (isWarmupID(message.getId())) {
            return;
        }

        int priority = readTruePriority(message);

        if (isValidPriority(priority)) {
            this.started[priority]++;
        }
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

        int priority = readTruePriority(message);

        if (!isValidPriority(priority)) {
            return;
        }

        this.relayed[priority]++;

        if (finalTarget) {
            this.delivered[priority]++;

            double latency =
                    getSimTime() - message.getCreationTime();

            this.latencies[priority].add(
                    Double.valueOf(latency));

            int hopCount = message.getHops().size() - 1;

            if (hopCount < 0) {
                hopCount = 0;
            }

            this.hopCounts[priority].add(
                    Integer.valueOf(hopCount));
        }
    }

    @Override
    public void messageTransferAborted(
            Message message,
            DTNHost from,
            DTNHost to) {

        if (isWarmupID(message.getId())) {
            return;
        }

        int priority = readTruePriority(message);

        if (isValidPriority(priority)) {
            this.aborted[priority]++;
        }
    }

    @Override
    public void messageDeleted(
            Message message,
            DTNHost where,
            boolean wasDropped) {

        if (isWarmupID(message.getId())) {
            return;
        }

        int priority = readTruePriority(message);

        if (!isValidPriority(priority)) {
            return;
        }

        if (wasDropped) {
            this.dropped[priority]++;
        }
        else {
            this.removed[priority]++;
        }
    }

    @Override
    public void done() {
        write(
            "True-priority message statistics for scenario " +
            getScenarioName()
        );

        write("sim_time: " + format(getSimTime()));

        write(
            "messages_without_true_priority: " +
            this.messagesWithoutTruePriority
        );

        write("");

        write(
            "priority created delivered delivery_prob " +
            "started relayed overhead_ratio aborted " +
            "dropped removed latency_avg latency_med " +
            "hopcount_avg hopcount_med"
        );

        for (int priority = MIN_PRIORITY;
                priority <= MAX_PRIORITY;
                priority++) {

            double deliveryProbability =
                    ratio(
                            this.delivered[priority],
                            this.created[priority]
                    );

            double overheadRatio;

            if (this.delivered[priority] == 0) {
                overheadRatio = Double.NaN;
            }
            else {
                overheadRatio =
                        (
                            this.relayed[priority] -
                            this.delivered[priority]
                        ) /
                        (double) this.delivered[priority];
            }

            double averageLatency =
                    averageDoubles(this.latencies[priority]);

            double medianLatency =
                    medianDoubles(this.latencies[priority]);

            double averageHopCount =
                    averageIntegers(this.hopCounts[priority]);

            double medianHopCount =
                    medianIntegers(this.hopCounts[priority]);

            write(
                "P" + priority + " " +
                this.created[priority] + " " +
                this.delivered[priority] + " " +
                format(deliveryProbability) + " " +
                this.started[priority] + " " +
                this.relayed[priority] + " " +
                format(overheadRatio) + " " +
                this.aborted[priority] + " " +
                this.dropped[priority] + " " +
                this.removed[priority] + " " +
                format(averageLatency) + " " +
                format(medianLatency) + " " +
                format(averageHopCount) + " " +
                format(medianHopCount)
            );
        }

        super.done();
    }

    private int readTruePriority(Message message) {
        Object value =
                message.getProperty(TRUE_PRIORITY_PROPERTY);

        if (value instanceof Number) {
            int priority =
                    ((Number) value).intValue();

            return isValidPriority(priority)
                    ? priority
                    : -1;
        }

        if (value instanceof String) {
            String text =
                    ((String) value)
                    .trim()
                    .toUpperCase();

            if (text.startsWith("P")) {
                text = text.substring(1);
            }

            try {
                int priority = Integer.parseInt(text);

                return isValidPriority(priority)
                        ? priority
                        : -1;
            }
            catch (NumberFormatException error) {
                return -1;
            }
        }

        return -1;
    }

    private boolean isValidPriority(int priority) {
        return priority >= MIN_PRIORITY &&
                priority <= MAX_PRIORITY;
    }

    private double ratio(int numerator, int denominator) {
        if (denominator == 0) {
            return Double.NaN;
        }

        return numerator / (double) denominator;
    }

    private double averageDoubles(List<Double> values) {
        if (values.isEmpty()) {
            return Double.NaN;
        }

        double total = 0.0;

        for (Double value : values) {
            total += value.doubleValue();
        }

        return total / values.size();
    }

    private double medianDoubles(List<Double> values) {
        if (values.isEmpty()) {
            return Double.NaN;
        }

        List<Double> sorted =
                new ArrayList<Double>(values);

        Collections.sort(sorted);

        int middle = sorted.size() / 2;

        if (sorted.size() % 2 == 1) {
            return sorted.get(middle).doubleValue();
        }

        return (
                sorted.get(middle - 1).doubleValue() +
                sorted.get(middle).doubleValue()
            ) / 2.0;
    }

    private double averageIntegers(List<Integer> values) {
        if (values.isEmpty()) {
            return Double.NaN;
        }

        double total = 0.0;

        for (Integer value : values) {
            total += value.intValue();
        }

        return total / values.size();
    }

    private double medianIntegers(List<Integer> values) {
        if (values.isEmpty()) {
            return Double.NaN;
        }

        List<Integer> sorted =
                new ArrayList<Integer>(values);

        Collections.sort(sorted);

        int middle = sorted.size() / 2;

        if (sorted.size() % 2 == 1) {
            return sorted.get(middle).doubleValue();
        }

        return (
                sorted.get(middle - 1).doubleValue() +
                sorted.get(middle).doubleValue()
            ) / 2.0;
    }
}