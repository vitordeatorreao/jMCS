/* DynamicVotingSelectionDES.java
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

import br.ufpe.cin.vat.jmcs.utils.Enumerables;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Implementation of the Dynamic Voting with Selection (DVS) for dynamically
 * selecting an sub ensemble of classifier for a given test instance.
 * @author vitordeatorreao
 * @since 0.1
 *
 */
public class DynamicVotingSelectionDES extends DynamicVoting
{
    @Override
    public Classifier[] selectClassifiers(Instance testInstance)
            throws Exception
    {
        Classifier[] pool = this.getClassifiers();
        int n_neighbors = this.getKNeighbors();
        Instances neighbors = this.getKnnAlgorithm()
                                  .kNearestNeighbours(testInstance,
                                                      n_neighbors);
        SelectionResult result = this.calculateErrorWeights(pool, neighbors,
                                                            n_neighbors);
        Double[] weights = new Double[pool.length];
        for (int j = 0; j < pool.length; j++) {
            weights[j] = result.weightedErrors[j];
        }
        // Get the lower half with the least error
        int[] indexes = Enumerables.SortIndexes(weights);
        int halfSize = pool.length / 2;
        double[] weightedErrors = new double[halfSize];
        double sum = 0.0;
        Classifier[] ensemble = new Classifier[halfSize];
        for (int j = 0; j < halfSize; j++) {
            weightedErrors[j] = result.weightedErrors[indexes[j]];
            ensemble[j] = pool[indexes[j]];
            sum += weightedErrors[j];
        }
        if (sum > 0) {
            Utils.normalize(weightedErrors, sum);
        }
        double[] classifierWeights = new double[halfSize];
        for (int j = 0; j < halfSize; j++) {
            classifierWeights[j] = 1 - weightedErrors[j];
        }
        this.combiner.setClassifiersWeights(classifierWeights);
        return ensemble;
    }
}
