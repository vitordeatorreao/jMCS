/* KnnAccuracyBasedDCS.java
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

import br.ufpe.cin.vat.jmcs.utils.Labels;
import weka.classifiers.Classifier;
import weka.core.Instance;

/**
 * Abstract class with the common methods for the Dynamic Classifier Selection
 * approaches that are based on nearest neighbor search and accuracy, such as
 * Overall Local Accuracy (OLA) and Local Class Accuracy (LCA).
 * @author vitordeatorreao
 * @since 0.1
 *
 */
public abstract class KnnAccuracyBasedDCS extends NearestNeighborsBasedDS
    implements DynamicClassifierSelection
{
    /**
     * Selects a classifier to label the test instance according to specific
     * selection approach.
     * @param testInstance - The instance to be labeled.
     * @param classifiers - The original pool of classifiers from which to
     * select one.
     * @return The index of the best classifier according to the selection rule.
     * @throws Exception - In case any of the classifiers in the pool have any
     * trouble classifying the nearest neighbors instances or the test instance.
     * @since 0.1 
     */
    protected abstract int selectClassifier(
            Instance testInstance, Classifier[] classifiers) throws Exception;
    
    @Override
    public Classifier selectClassifier(Instance testInstance) throws Exception
    {
        Classifier[] classifiers = this.getClassifiers();
        int index = this.selectClassifier(testInstance, classifiers);
        return classifiers[index];
    }

    @Override
    public double classifyInstance(Instance testInstance) throws Exception
    {
        Classifier[] classifiers = this.getClassifiers();
        if (classifiers.length < 1) {
            throw new UnsupportedOperationException(
                "Cannot classify this instance with an empty pool.");
        }
        double[] labels = new double[classifiers.length];
        labels[0] = classifiers[0].classifyInstance(testInstance);
        boolean unanimous = true;
        for (int i = 1; i < classifiers.length; i++) {
            labels[i] = classifiers[i].classifyInstance(testInstance);
            if (!Labels.Equals(labels[i], labels[i - 1])) unanimous = false;
        }
        // if all the classifiers agree, then return the label
        if (unanimous) return labels[0];
        // else, get the k nearest neighbors
        int chosenIndex = this.selectClassifier(testInstance, classifiers);
        return labels[chosenIndex];
    }

    @Override
    public double[] distributionForInstance(Instance testInstance)
            throws Exception
    {
        Classifier[] classifiers = this.getClassifiers();
        int chosenIndex = this.selectClassifier(testInstance, classifiers);
        return classifiers[chosenIndex].distributionForInstance(testInstance);
    }
}
