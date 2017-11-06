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

import mulan.classifier.MultiLabelLearner;
import weka.classifiers.Classifier;
import weka.classifiers.MultipleClassifiersCombiner;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Abstract utility class for handling settings common to Dynamic Ensemble
 * Selection techniques based on the use of multi-label classifiers as
 * meta-classifiers.
 * @author vitordeatorreao
 * @since 0.1
 *
 */
public class MultiLabelDynamicEnsembleSelection
    implements DynamicSelection, DynamicEnsembleSelection, Classifier
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
     * Constructs a new instance of MultiLabelDynamicEnsembleSelection with
     * clean settings.
     * @since 0.1 
     */
    public MultiLabelDynamicEnsembleSelection() { }

    /**
     * Constructs a new instance of MultiLabelDynamicEnsembleSelection with
     * the given array of classifiers as the original pool of classifiers.
     * @param classifiers - The original pool of classifiers from which a
     * subset will be selected at each classification step.
     * @since 0.1
     */
    public MultiLabelDynamicEnsembleSelection(Classifier[] classifiers)
    {
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
    public MultiLabelDynamicEnsembleSelection(MultipleClassifiersCombiner ensemble)
    {
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
        this.classifiers = classifiers;
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

    public void buildClassifier(Instances selectionDataSet) throws Exception
    {
        // TODO Auto-generated method stub
        return;
    }

    public double classifyInstance(Instance instance) throws Exception
    {
        // TODO Auto-generated method stub
        return 0;
    }

    public double[] distributionForInstance(Instance instance) throws Exception
    {
        // TODO Auto-generated method stub
        return null;
    }

    public Capabilities getCapabilities()
    {
        // TODO Auto-generated method stub
        return null;
    }
}
