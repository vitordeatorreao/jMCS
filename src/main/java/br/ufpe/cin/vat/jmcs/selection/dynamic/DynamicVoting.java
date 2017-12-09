/* DynamicVotingDES.java
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

import br.ufpe.cin.vat.jmcs.combination.WeightedVote;
import br.ufpe.cin.vat.jmcs.utils.Labels;
import weka.classifiers.Classifier;
import weka.classifiers.MultipleClassifiersCombiner;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Implementation of the Dynamic Voting (DV) fusion approach. It can't be
 * considered a Dynamic Ensemble Selection technique, because it doesn't select
 * a subset of the original pool. Instead, it uses the entire pool, but combines
 * them with different weights. However, for making code this class easier, we
 * are implementing it as an extension of DS classes.
 * @author vitordeatorreao
 * @since 0.1
 *
 */
public class DynamicVoting extends NearestNeighborsBasedDS
        implements DynamicEnsembleSelection
{
    protected WeightedVote combiner;

    public DynamicVoting()
    {
        this.combiner = new WeightedVote();
    }

    public DynamicVoting(int kNeighbors)
    {
        super(kNeighbors);
        this.combiner = new WeightedVote();
    }

    @Override
    public void setCombiner(MultipleClassifiersCombiner combiner)
    {
        return; // you can't set the combiner
    }

    @Override
    public MultipleClassifiersCombiner getCombiner()
    {
        return this.combiner;
    }

    protected class SelectionResult
    {
        public double[] distances;
        public double[] weightedErrors;
        public double sum;
    }

    protected SelectionResult calculateErrorWeights(Classifier[] pool,
            Instances neighbors, int n_neighbors) throws Exception
    {
        SelectionResult result = new SelectionResult();
        result.distances = this.getKnnAlgorithm().getDistances();
        result.weightedErrors = new double[pool.length];
        result.sum = 0.0;
        for (int j = 0; j < pool.length; j++) {
            result.weightedErrors[j] = 0.0;
            for (int i = 0; i < n_neighbors; i++) {
                Instance neighbor = neighbors.get(i);
                double answer = pool[j].classifyInstance(neighbor);
                double error = Labels.Equals(answer, neighbor.classValue()) ?
                               0.0 : 1.0;
                result.weightedErrors[j] += result.distances[i] * error;
            }
            result.weightedErrors[j] /= n_neighbors > 0 ? n_neighbors : 1.0;
            result.sum += result.weightedErrors[j];
        }
        return result;
    }

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
        if (result.sum > 0) {
            Utils.normalize(result.weightedErrors, result.sum);
        }
        double[] classifierWeights = new double[pool.length];
        for (int j = 0; j < pool.length; j++) {
            classifierWeights[j] = 1 - result.weightedErrors[j];
        }
        this.combiner.setClassifiersWeights(classifierWeights);
        return pool;
    }

    @Override
    public double classifyInstance(Instance testInstance) throws Exception
    {
        this.combiner.setClassifiers(this.selectClassifiers(testInstance));
        return this.combiner.classifyInstance(testInstance);
    }

    @Override
    public double[] distributionForInstance(Instance testInstance)
            throws Exception
    {
        this.combiner.setClassifiers(this.selectClassifiers(testInstance));
        return this.combiner.distributionForInstance(testInstance);
    }

}
