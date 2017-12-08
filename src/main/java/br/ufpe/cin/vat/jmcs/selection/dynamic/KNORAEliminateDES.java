/* KNORAEliminateDES.java
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

import weka.classifiers.Classifier;
import weka.classifiers.MultipleClassifiersCombiner;
import weka.classifiers.meta.Vote;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Implementation of the KNORA Eliminate approach for dynamically selecting a
 * sub ensemble of classifiers for a given test instance.
 * @author vitordeatorreao
 * @since 0.1
 *
 */
public class KNORAEliminateDES extends NearestNeighborsBasedDS
    implements DynamicEnsembleSelection
{
    /**
     * The way the selected sub ensemble should be combined to form the final
     * answer. Major Vote by default.
     */
    private MultipleClassifiersCombiner combiner;

    /**
     * Creates a new Instance of KNORA Eliminate with the default parameters.
     */
    public KNORAEliminateDES()
    {
        super();
        this.combiner = new Vote();
    }

    /**
     * Creates a new Instance of KNORA Eliminate with the given number of
     * neighbors in the search. The remaining parameters are set to their
     * default values.
     * @param kNeighbors - Number of nearest neighbors to be searched.
     */
    public KNORAEliminateDES(int kNeighbors)
    {
        super(kNeighbors);
        this.combiner = new Vote();
    }

    /**
     * Creates a new Instance of KNORA Eliminate with the given way of combining
     * the selected sub ensemble output.
     * @param combiner - A WEKA way for combining multiple classifier`s output.
     */
    public KNORAEliminateDES(MultipleClassifiersCombiner combiner)
    {
        super();
        this.combiner = combiner;
    }

    /**
     * Creates a new Instance of KNORA Eliminate with the given number of
     * neighbors for the search and the given way of combining the selected sub
     * ensemble output.
     * @param kNeighbors - Number of nearest neighbors to be searched.
     * @param combiner - A WEKA way for combining multiple classifier`s output.
     */
    public KNORAEliminateDES(int kNeighbors,
            MultipleClassifiersCombiner combiner)
    {
        super(kNeighbors);
        this.combiner = combiner;
    }

    @Override
    public void setCombiner(MultipleClassifiersCombiner combiner) {
        this.combiner = combiner;
    }

    @Override
    public MultipleClassifiersCombiner getCombiner() {
        return this.combiner;
    }

    @Override
    public Classifier[] selectClassifiers(Instance testInstance)
            throws Exception {
        Classifier[] pool = this.getClassifiers();
        List<Classifier> selected = new ArrayList<Classifier>();
        for (int k = this.getKNeighbors(); k > 0; k--) {
            Instances neighbors = this.getKnnAlgorithm()
                                      .kNearestNeighbours(testInstance, k);
            for (int i = 0; i < pool.length; i++) {
                boolean toAdd = true;
                for (Instance neighbor : neighbors) {
                    if (pool[i].classifyInstance(neighbor) !=
                        neighbor.classValue()) {
                        toAdd = false;
                        break;
                    }
                }
                if (toAdd) selected.add(pool[i]);
            }
            if (!selected.isEmpty()) break;
        }
        Classifier[] aux = new Classifier[selected.size()];
        return selected.toArray(aux);
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
