/* MCBBasedDCS.java
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

import br.ufpe.cin.vat.jmcs.utils.Enumerables;
import br.ufpe.cin.vat.jmcs.utils.Labels;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Implementation of the MCB based dynamic classifier selection technique
 * proposed by Giacinto & Roli (2000).
 * @author vitordeatorreao
 * @since 0.1
 *
 */
public class MCBBasedDCS extends KnnAccuracyBasedDCS
{
    /**
     * The threshold for deciding if a given neighbor is included in the final
     * neighborhood.
     */
    private double similarityThreshold;

    /**
     * Creates a new instance of the MCB based DCS technique with default value
     * for the similarity threshold.
     */
    public MCBBasedDCS()
    {
        super();
        this.similarityThreshold = 0.5; 
    }

    /**
     * Creates a new instance of the MCB based DCS technique with the given
     * value configured as similarity threshold.
     * @param similarityThreshold - The similarity threshold with which the
     * created instance should be configured.
     */
    public MCBBasedDCS(double similarityThreshold)
    {
        super();
        this.similarityThreshold = similarityThreshold;
    }

    /**
     * Configures the similarity threshold to the given double amount.
     * @param threshold - The new similarity threshold for deciding which
     * neighbors should be a part of the final neighborhood.
     */
    public void setSimilarityThreshold(double threshold)
    {
        this.similarityThreshold = threshold;
    }

    /**
     * Retrieves the configured similarity threshold.
     * @return The current value for the similarity threshold.
     */
    public double getSimilarityThreshold()
    {
        return this.similarityThreshold;
    }

    @Override
    protected int selectClassifier(Instance testInstance,
            Classifier[] classifiers) throws Exception
    {
        int n_neighbors = this.getKNeighbors(); 
        Instances neighbors = this.getKnnAlgorithm()
                                  .kNearestNeighbours(testInstance,
                                                      n_neighbors);
        double[] testMCB = new double[classifiers.length];
        for (int j = 0; j < classifiers.length; j++) {
            testMCB[j] = classifiers[j].classifyInstance(testInstance);
        }
        List<Instance> finalNeighborhood = new ArrayList<Instance>();
        for (Instance neighbor : neighbors) {
            double[] neighborMCB = new double[classifiers.length];
            for (int j = 0; j < classifiers.length; j++) {
                neighborMCB[j] = classifiers[j].classifyInstance(neighbor);
            }
            if (similarity(testMCB, neighborMCB) > this.similarityThreshold) {
                finalNeighborhood.add(neighbor);
            }
        }
        Integer[] correctAnswerCount = new Integer[classifiers.length];
        for (int i = 0; i < classifiers.length; i++) {
            correctAnswerCount[i] = new Integer(0);
            for (Instance neighbor : finalNeighborhood) {
                double answer = classifiers[i].classifyInstance(neighbor);
                if (Labels.Equals(answer, neighbor.classValue())) {
                    correctAnswerCount[i]++;
                }
            }
        }
        return Enumerables.MaxIndex(correctAnswerCount);
    }

    /**
     * Calculates the similarity between two database instances based on the
     * Multiple Classifier Behavior (MCB) of the two. 
     * @param mcb1 - The first instance's MCB.
     * @param mcb2 - The second instance's MCB.
     * @return The value of the similarity between the two MCBs as defined in
     * Giacinto & Roli (2000).
     */
    protected double similarity(double[] mcb1, double[] mcb2)
    {
        if (mcb1.length != mcb2.length) {
            throw new IllegalArgumentException("Both MCBs must have the same " +
                                               "length.");
        }
        double sum = 0.0;
        for (int i = 0; i < mcb1.length; i++) {
            sum += Labels.Equals(mcb1[i], mcb2[i]) ? 1.0 : 0.0;
        }
        return sum / mcb1.length;
    }
    
}
