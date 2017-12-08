/* DynamicSelectionDCS.java
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
 * Implementation of <b>the</b> Dynamic Selection algorithm proposed by Puuronen
 * et al (1999).
 * @author vitordeatorreao
 * @since 0.1
 *
 */
public class DynamicSelectionDCS extends KnnAccuracyBasedDCS
{
    @Override
    protected int selectClassifier(Instance testInstance,
            Classifier[] classifiers) throws Exception
    {
        int n_neighbors = this.getKNeighbors();
        Instances neighbors = this.getKnnAlgorithm()
                                  .kNearestNeighbours(testInstance,
                                                      n_neighbors);
        double[] distances = this.getKnnAlgorithm().getDistances();
        Double[] weightedErrors = new Double[classifiers.length];
        for (int j = 0; j < classifiers.length; j++) {
            weightedErrors[j] = new Double(0.0);
            for (int i = 0; i < n_neighbors; i++) {
                Instance neighbor = neighbors.get(i);
                double answer = classifiers[j].classifyInstance(neighbor);
                double error = Labels.Equals(answer, neighbor.classValue()) ?
                               0.0 : 1.0;
                weightedErrors[j] += distances[i] * error;
            }
            weightedErrors[j] /= n_neighbors > 0 ? n_neighbors : 1.0;
        }
        return Enumerables.MinIndex(weightedErrors);
    }
}
