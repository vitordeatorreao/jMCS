/* OverallLocalAccuracyDCS.java
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
 * Implementation of the Overall Local Accuracy (OLA) approach for dynamically
 * selecting a classifier for a given test instance.
 * @author vitordeatorreao
 * @since 0.1
 *
 */
public class OverallLocalAccuracyDCS extends KnnAccuracyBasedDCS
{
    @Override
    protected int selectClassifier(Instance testInstance,
            Classifier[] classifiers) throws Exception

    {
        int n_neighbors = this.getKNeighbors(); 
        Instances neighbors = this.getKnnAlgorithm()
                                  .kNearestNeighbours(testInstance,
                                                      n_neighbors);
        Integer[] correctAnswerCount = new Integer[classifiers.length];
        for (int i = 0; i < classifiers.length; i++) {
            correctAnswerCount[i] = new Integer(0);
            for (Instance neighbor : neighbors) {
                double answer = classifiers[i].classifyInstance(neighbor);
                if (Labels.Equals(answer, neighbor.classValue())) {
                    correctAnswerCount[i]++;
                }
            }
        }
        return Enumerables.MaxIndex(correctAnswerCount);
    }
}
