/* NearestNeighborsBasedDS.java
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

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.BallTree;
import weka.core.neighboursearch.NearestNeighbourSearch;

/**
 * Abstract utility class containing common methods for nearest neighbor based
 * dynamic selection approaches (both of ensemble and classifier).
 * @author vitordeatorreao
 * @since 0.1
 *
 */
public abstract class NearestNeighborsBasedDS
    implements DynamicSelection
{
    /**
     * The number of nearest neighbors to be considered by the approach.
     * @since 0.1 
     */
    private int kNeighbors;

    /**
     * The set of classifiers in the original pool.
     */
    private Classifier[] classifiers;
    
    private NearestNeighbourSearch knn;

    /**
     * Constructs a new instance with the default parameters.
     * @since 0.1
     */
    public NearestNeighborsBasedDS() {
        this.kNeighbors = 10;
        this.classifiers = new Classifier[0];
    }

    /**
     * Creates a new instance with the given number of nearest neighbors
     * configured.
     * @param kNeighbors
     * @since 0.1
     */
    public NearestNeighborsBasedDS(int kNeighbors) {
        this.kNeighbors = kNeighbors;
        this.classifiers = new Classifier[0];
    }

    /**
     * Retrieves the number of neighbors considered by the approach.
     * @return The number of nearest neighbors that will be considered
     * for selecting the classifier or sub-ensemble.
     * @since 0.1
     */
    public int getKNeighbors() {
        return this.kNeighbors;
    }

    /**
     * Configures the number of neighbors to be used by the approach. 
     * @param k - The number of neighbors to be used.
     * @since 0.1
     */
    public void setKNeighbors(int k) {
        this.kNeighbors = k;
    }
    
    public NearestNeighbourSearch getKnnAlgorithm() {
        return this.knn;
    }

    @Override
    public void setClassifiers(Classifier[] classifiers) {
        this.classifiers = classifiers;
    }

    @Override
    public Classifier[] getClassifiers() {
        return this.classifiers;
    }

    @Override
    public void buildClassifier(Instances selectionInstances) throws Exception
    {
        this.knn = new BallTree(selectionInstances);
    }

    @Override
    public abstract double classifyInstance(Instance arg0) throws Exception;

    @Override
    public abstract double[] distributionForInstance(Instance arg0)
        throws Exception;

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException(
                "getCapabilities is not implemented for the class " +
                NearestNeighborsBasedDS.class.getName());
    }
}
