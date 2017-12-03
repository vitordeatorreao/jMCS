/* MultiLabelDynamicEnsembleSelection.java
 * Copyright (C) 2017  Vitor de Albuquerque Torreao
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package br.ufpe.cin.vat.jmcs.selection.dynamic;

import java.util.ArrayList;
import java.util.List;

import br.ufpe.cin.vat.jmcs.utils.MultiLabel;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.LabelsMetaData;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.classifiers.MultipleClassifiersCombiner;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Utility class for handling settings common to Dynamic Ensemble
 * Selection techniques based on the use of multi-label classifiers as
 * meta-classifiers.
 * @author vitordeatorreao
 * @since 0.1
 *
 */
public class MultiLabelDynamicEnsembleSelection
    implements Classifier, DynamicSelection, DynamicEnsembleSelection
{
    /**
     * The set of classifiers in the original pool. 
     */
    private Classifier[] classifiers;

    /**
     * The instance used to combine the answers from the selected subset of
     * classifiers. 
     */
    private MultipleClassifiersCombiner combiner;

    /**
     * The instance of a multi-label algorithm used for selecting the subset of
     * ensembles for the classification tasks.
     */
    private MultiLabelLearner mlAlgorithm;

    /**
     * The multi-label data set for the selection task. 
     */
    private MultiLabelInstances selectionDataSet;

    /**
     * Threshold used in the MultiLabelOutput to form a bipartition.
     */
    private Double threshold;

    /**
     * Holds the attribute information for the current multilabel problem
     */
    private ArrayList<Attribute> attributeInfo;

    /**
     * Constructs a new instance of MultiLabelDynamicEnsembleSelection with
     * clean settings.
     * @since 0.1 
     */
    public MultiLabelDynamicEnsembleSelection()
    {
        this.threshold = null;
        this.classifiers = new Classifier[0];
    }

    /**
     * Constructs a new instance of MultiLabelDynamicEnsembleSelection with
     * the given array of classifiers as the original pool of classifiers.
     * @param classifiers - The original pool of classifiers from which a
     * subset will be selected at each classification step.
     * @since 0.1
     */
    public MultiLabelDynamicEnsembleSelection(Classifier[] classifiers)
    {
        this();
        this.classifiers = classifiers;
    }

    /**
     * Constructs a new instance of MultiLabelDynamicEnsembleSelection with
     * the given combiner set.
     * @param ensemble - The combiner to be used to fuse the answers of the
     * selected classifiers. The classifiers in the combiner will be set as
     * the original pool of classifiers.
     * @since 0.1
     */
    public MultiLabelDynamicEnsembleSelection(
            MultipleClassifiersCombiner ensemble)
    {
        this();
        this.classifiers = ensemble.getClassifiers();
        this.combiner = ensemble;
    }

    /**
     * Construct a new instance of MultiLabelDynamicEnsembleSelection with
     * the given list of classifiers as the original pool from which to
     * select the final ensemble, and with the given combiner set.
     * @param classifiers - The original pool of classifiers from which a
     * subset will be selected at each classification step.
     * @param combiner - The combiner to be used to fuse the answers of the
     * selected classifiers.
     * @since 0.1
     */
    public MultiLabelDynamicEnsembleSelection(
            Classifier[] classifiers, MultipleClassifiersCombiner combiner)
    {
        this();
        this.classifiers = classifiers;
        this.combiner = combiner;
    }

    /**
     * Retrieves the multi-label algorithm configured to be used in the dynamic
     * selection in each classification of a test instance.
     * @return The multi-label algorithm used for selection.
     * @since 0.1
     */
    public MultiLabelLearner getMultiLabelAlgorithm()
    {
        return this.mlAlgorithm;
    }

    /**
     * Configures the multi-label algorithm to be used in the selection step.
     * @param mlAlgothm - The multi-label algorithm to be used in the
     * selection.
     * @since 0.1
     */
    public void setMultiLabelAlgorithm(MultiLabelLearner mlAlgothm)
    {
        this.mlAlgorithm = mlAlgothm;
    }

    /**
     * Retrieves the configured threshold. It may be null if no threshold was
     * configured.
     * @return The configured threshold or null.
     */
    public Double getThreshold()
    {
        return this.threshold;
    }

    /**
     * Configures the threshold to be used in order to get the bipartition.
     * @param threshold - The threshold to be used.
     * @since 0.1
     */
    public void setThreshold(double threshold)
    {
        this.threshold = threshold;
    }

    public Classifier[] getClassifiers()
    {
        return this.classifiers;
    }

    public void setClassifiers(Classifier[] classifiers)
    {
        this.classifiers = classifiers;
    }

    public MultipleClassifiersCombiner getCombiner()
    {
        return this.combiner;
    }

    public void setCombiner(MultipleClassifiersCombiner combiner)
    {
        this.combiner = combiner;
    }

    /**
     * Constructs a new data set by adding the classifiers' outputs as new
     * columns. 
     * @param selectionDataSet - The data set used as meta-information for
     * the dynamic selection step.
     * @return The new data set constructed by adding the classifiers' outputs
     * as new columns.
     * @throws Exception - Throws up any exception that occurs during the
     * execution of the classifiers in the original ensemble.
     * @since 0.1
     */
    public Instances getMultiLabelDataSet(Instances selectionDataSet)
            throws Exception
    {
        int numAttributes = selectionDataSet.numAttributes();
        int numLabels = this.classifiers.length;
        int numInstances = selectionDataSet.size();
        // -1 due to the class label
        int initialCapacity = numAttributes - 1 + numLabels;
        this.attributeInfo = new ArrayList<Attribute>(initialCapacity);
        for (int i = 0; i < numAttributes; i++)
        {
            if (i == selectionDataSet.classIndex()) continue;
            this.attributeInfo.add(selectionDataSet.attribute(i));
        }
        List<String> multiLabelOutput = new ArrayList<String>(2);
        multiLabelOutput.add("0");
        multiLabelOutput.add("1");
        for (int c = 0; c < numLabels; c++)
        {
            this.attributeInfo.add(
                    new Attribute("Classifier" + c, multiLabelOutput));
        }
        Instances aux = new Instances(
                "Selection", this.attributeInfo, numInstances);
        for (int n = 0; n < numInstances; n++)
        {
            Instance data = new DenseInstance(initialCapacity);
            for (int a = 0; a < numAttributes - 1; a++)
            {
                data.setValue(aux.attribute(a),
                              selectionDataSet.get(n).value(a));
            }
            int start = numAttributes - 1;
            for (int c = 0; c < numLabels; c++)
            {
                Instance instance = selectionDataSet.get(n);
                double correct = instance.classValue();
                double actual = this.classifiers[c].classifyInstance(instance);
                double delta = Math.abs(actual - correct); 
                if (delta < 0.0001)
                {
                    data.setValue(aux.attribute(c + start), "1");
                }
                else
                {
                    data.setValue(aux.attribute(c + start), "0");
                }
            }
            aux.add(data);
        }
        return aux;
    }

    /**
     * Retrieves a bipartition of classifiers that are competent or not for
     * classifying the given Instance. It will either use the multi-label
     * algorithm's default bipartition strategy or the OneThreshold strategy,
     * depending on whether the Threshold has been set or not.
     * @param instance - The Instance to be classified.
     * @return The bipartition of classifiers that are competent or not for
     * classifying the given Instance.
     * @throws Exception In case the multi-label model cannot classify the
     * given instance.
     */
    public boolean[] getBipartition(Instance instance) throws Exception
    {
        instance = this.getMultiLabelInstanceForClassification(instance);
        MultiLabelOutput output = this.mlAlgorithm.makePrediction(instance);
		boolean[] bipartition;
		// use default, or OneThreshold strategy to form a bipartition
		if (output.hasConfidences() && this.threshold != null)
		{
			bipartition = new boolean[this.classifiers.length];
			double[] confidence = output.getConfidences();
			for (int c = 0; c < this.classifiers.length; c++)
			{
				if (confidence[c] > this.threshold) bipartition[c] = true;
				else bipartition[c] = false;
			}
		}
		else
		{
			bipartition = output.getBipartition();
		}
		return bipartition;
    }

	public double classifyInstance(Instance instance) throws Exception
	{
		this.configureCombiner(instance);
		return this.combiner.classifyInstance(instance);
	}

    public void buildClassifier(Instances selectionDataSet) throws Exception
    {
        if (this.classifiers == null)
        {
            throw new IllegalStateException("You can't call buildSelector " +
                "before configuring the initial pool.");
        }
        if (selectionDataSet.classIndex() < 0)
        {
            throw new IllegalArgumentException("The provided selection data "+
                "must have a class label configured.");
        }
        int numAttributes = selectionDataSet.numAttributes();
        // discount for the class
        if (selectionDataSet.classIndex() > 0) numAttributes--;
        Instances multiLabelDataSet = getMultiLabelDataSet(selectionDataSet);
        int[] labelsIndexes = new int[this.classifiers.length];
        for (int c = 0; c < this.classifiers.length; c++)
        {
            labelsIndexes[c] = c + numAttributes;
        }
        LabelsMetaData metaData =
                MultiLabel.getLabelsMetaData(multiLabelDataSet, labelsIndexes);
        this.selectionDataSet = new MultiLabelInstances(multiLabelDataSet,
                                                        metaData);
        this.mlAlgorithm.build(this.selectionDataSet);
    }

    public double[] distributionForInstance(Instance instance) throws Exception
    {
        this.configureCombiner(instance);
        return this.combiner.distributionForInstance(instance);
    }

    public Capabilities getCapabilities()
    {
        throw new UnsupportedOperationException(
            "getCapabilities is not implemented for the class " +
            MultiLabelDynamicEnsembleSelection.class.getName());
    }

    /**
     * Calls the combiner's setClassifiers with a list of classifiers in the
     * MultiLabel algorithm's bipartition.
     * @param instance - The instance being classified. It is used to get the
     * bipartition.
     * @throws Exception - Thrown by getBipartition.
     * @since 0.1
     */
    private void configureCombiner(Instance instance) throws Exception
    {
        boolean[] bipartition = this.getBipartition(instance);
        List<Classifier> classifiers = new ArrayList<Classifier>();
        for (int c = 0; c < this.classifiers.length; c++)
        {
            if (bipartition[c]) classifiers.add(this.classifiers[c]);
        }
        Classifier[] aux = new Classifier[classifiers.size()];
        this.combiner.setClassifiers(classifiers.toArray(aux));
    }

    /**
     * When the instance comes in for classification, it doesn't have all the
     * needed attributes, since it lacks the multiple labels. This helper
     * method adds those attributes, so you can pass the instance to the
     * multilabel algorithm's makePrediction method.
     * @param instance - The instance being classified.
     * @return The Instance ready for use in makePrediction.
     * @since 0.1
     */
    private Instance getMultiLabelInstanceForClassification(Instance instance)
    {
        Instances aux = new Instances("Phony", this.attributeInfo, 1);
        Instance data = new DenseInstance(this.attributeInfo.size());
        int numAttrs = this.attributeInfo.size() - this.classifiers.length;
        for (int a = 0; a < numAttrs; a++)
        {
            data.setValue(aux.attribute(a),
                          instance.value(a));
        }
        return data;
    }
}
