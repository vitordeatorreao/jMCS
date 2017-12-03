/* LocalClassAccuracyDCS.java
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
import br.ufpe.cin.vat.jmcs.utils.Labels;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Implementation of the Local Class Accuracy (LCA) approach for dynamically
 * selecting a classifier for a given test instance.
 * @author vitordeatorreao
 * @since 0.1
 *
 */
public class LocalClassAccuracyDCS extends KnnAccuracyBasedDCS
{
    @Override
    protected int selectClassifier(Instance testInstance,
            Classifier[] classifiers) throws Exception
    {
        int n_neighbors = this.getKNeighbors(); 
        Instances neighbors = this.getKnnAlgorithm()
                                  .kNearestNeighbours(testInstance,
                                                      n_neighbors);
        double[] testLabels = new double[classifiers.length];
        Double[] classAccuracy = new Double[classifiers.length];
        for (int i = 0; i < classifiers.length; i++) {
            testLabels[i] = classifiers[i].classifyInstance(testInstance);
            int total = 0;
            int corrects = 0;
            for (Instance neighbor : neighbors) {
                double predicted = classifiers[i].classifyInstance(neighbor);
                if (Labels.Equals(testLabels[i], predicted)) {
                    total++;
                    if (Labels.Equals(predicted, neighbor.classValue())) {
                        corrects++;
                    }
                }
            }
            classAccuracy[i] = total > 0 ? corrects / total : 0.0; 
        }
        return Enumerables.MaxIndex(classAccuracy);
    }
}
