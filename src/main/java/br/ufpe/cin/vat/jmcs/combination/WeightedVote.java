package br.ufpe.cin.vat.jmcs.combination;

import weka.classifiers.meta.Vote;
import weka.core.Instance;
import weka.core.SelectedTag;
import weka.core.Utils;

/**
 * Implementation of the weighted average voting combiner.
 * @author vitordeatorreao
 * @since 0.1
 *
 */
public class WeightedVote extends Vote
{
    /**
     * Generated serial version UID
     */
    private static final long serialVersionUID = -1159196600645007482L;
    
    protected double[] m_classifierWeights;
    protected double[] m_preBuiltClassifiersWeights;

    public WeightedVote()
    {
        this.m_classifierWeights = new double[0];
        this.m_preBuiltClassifiersWeights = new double[0];
    }

    /**
     * Configures the weight vector for the classifier (there is a different one
     * for the pre-built classifiers).
     * @param weights - The new vector of classifier weights. Make sure it has
     * enough elements.
     */
    public void setClassifiersWeights(double[] weights)
    {
        this.m_classifierWeights = this.adjustWeights(weights);
    }

    /**
     * Retrieves the weight vector for the classifiers (there is a different one
     * for the pre-built classifiers).
     * @return The weight vector for the classifiers.
     */
    public double[] getClassifiersWeights()
    {
        return this.m_classifierWeights;
    }

    /**
     * Configures the weight vector for the pre-built classifiers.
     * @param weights - The new vector of pre-built classifier weights. Make
     * sure it has enough elements.
     */
    public void setPreBuiltClassifiersWeight(double[] weights)
    {
        this.m_preBuiltClassifiersWeights = this.adjustWeights(weights);
    }

    /**
     * Retrieves the weight vector for the pre-built classifiers.
     * @return The weight vector for the pre-built classifiers.
     */
    public double[] getPreBuiltClassifiersWeight()
    {
        return this.m_preBuiltClassifiersWeights;
    }

    private double[] adjustWeights(double[] weights) {
        double[] newWeights = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            if (weights[i] < 0) {
                throw new IllegalArgumentException(
                        String.format("You can't provide a negative weight. " +
                                      "(%f < 0)", weights[i]));
            }
            newWeights[i] = 1 + weights[i]; // make it at least 1
        }
        return newWeights;
    }

    /**
     * Parses a given list of options.
     * <p/>
     * 
     * <!-- options-start --> Valid options are:
     * <p/>
     * 
     * <pre>
     * -P &lt;path to serialized classifier&gt;
     *  Full path to serialized classifier to include.
     *  May be specified multiple times to include
     *  multiple serialized classifiers. Note: it does
     *  not make sense to use pre-built classifiers in
     *  a cross-validation.
     * </pre>
     * 
     * <pre>
     * -print
     *  Print the individual models in the output
     * </pre>
     * 
     * <pre>
     * -S &lt;num&gt;
     *  Random number seed.
     *  (default 1)
     * </pre>
     * 
     * <pre>
     * -B &lt;classifier specification&gt;
     *  Full class name of classifier to include, followed
     *  by scheme options. May be specified multiple times.
     *  (default: "weka.classifiers.rules.ZeroR")
     * </pre>
     * 
     * <pre>
     * -output-debug-info
     *  If set, classifier is run in debug mode and
     *  may output additional info to the console
     * </pre>
     * 
     * <pre>
     * -do-not-check-capabilities
     *  If set, classifier capabilities are not checked before classifier is built
     *  (use with caution).
     * </pre>
     * 
     * <pre>
     * Options specific to classifier weka.classifiers.rules.ZeroR:
     * </pre>
     * 
     * <pre>
     * -output-debug-info
     *  If set, classifier is run in debug mode and
     *  may output additional info to the console
     * </pre>
     * 
     * <pre>
     * -do-not-check-capabilities
     *  If set, classifier capabilities are not checked before classifier is built
     *  (use with caution).
     * </pre>
     * 
     * <!-- options-end -->
     * 
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception
    {
        super.setOptions(options);
        // Weighted Vote has only one possible combination rule: average.
        setCombinationRule(new SelectedTag(AVERAGE_RULE, TAGS_RULES));
    }
    
    protected double[] distributionForInstanceAverage(Instance instance)
        throws Exception
    {
        double[] probs = new double[instance.numClasses()];

        double weightSum = 0;
        for (int i = 0; i < this.m_Classifiers.length; i++) {
            double[] dist = getClassifier(i).distributionForInstance(instance);
            if (!instance.classAttribute().isNumeric()
                || !Utils.isMissingValue(dist[0])) {
                double weight = i < this.m_classifierWeights.length ?
                                this.m_classifierWeights[i] : 1.0;
                for (int j = 0; j < dist.length; j++) {
                    probs[j] += dist[j];
                }
                weightSum += weight;
            }
        }

        for (int i = 0; i < this.m_preBuiltClassifiers.size(); i++) {
            double[] dist = this.m_preBuiltClassifiers
                                .get(i).distributionForInstance(instance);
            if (!instance.classAttribute().isNumeric()
                || !Utils.isMissingValue(dist[0])) {
                double weight = i < this.m_preBuiltClassifiersWeights.length ?
                                this.m_preBuiltClassifiersWeights[i] : 1.0;
                for (int j = 0; j < dist.length; j++) {
                    probs[j] += dist[j];
                }
                weightSum += weight;
            }
        }

        if (instance.classAttribute().isNumeric()) {
            if (weightSum == 0) {
                probs[0] = Utils.missingValue();
            } else {
                for (int j = 0; j < probs.length; j++) {
                    probs[j] /= weightSum;
                }
            }
        } else {
            // Should normalize "probability" distribution
            if (Utils.sum(probs) > 0) {
                Utils.normalize(probs);
            }
        }
        return probs;
    }
}
