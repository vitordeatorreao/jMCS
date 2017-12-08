package br.ufpe.cin.vat.jmcs.selection.dynamic;

import br.ufpe.cin.vat.jmcs.combination.WeightedVote;
import br.ufpe.cin.vat.jmcs.utils.Labels;
import weka.classifiers.Classifier;
import weka.classifiers.MultipleClassifiersCombiner;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class DynamicVotingDES extends NearestNeighborsBasedDS
        implements DynamicEnsembleSelection
{
    private WeightedVote combiner;

    public DynamicVotingDES()
    {
        this.combiner = new WeightedVote();
    }

    public DynamicVotingDES(int kNeighbors)
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

    @Override
    public Classifier[] selectClassifiers(Instance testInstance)
            throws Exception
    {
        Classifier[] pool = this.getClassifiers();
        int n_neighbors = this.getKNeighbors();
        Instances neighbors = this.getKnnAlgorithm()
                                  .kNearestNeighbours(testInstance,
                                                      n_neighbors);
        double[] distances = this.getKnnAlgorithm().getDistances();
        double[] weightedErrors = new double[pool.length];
        double sum = 0.0;
        for (int j = 0; j < pool.length; j++) {
            weightedErrors[j] = 0.0;
            for (int i = 0; i < n_neighbors; i++) {
                Instance neighbor = neighbors.get(i);
                double answer = pool[j].classifyInstance(neighbor);
                double error = Labels.Equals(answer, neighbor.classValue()) ?
                               0.0 : 1.0;
                weightedErrors[j] += distances[i] * error;
            }
            weightedErrors[j] /= n_neighbors > 0 ? n_neighbors : 1.0;
            sum += weightedErrors[j];
        }
        if (sum > 0) {
            Utils.normalize(weightedErrors, sum);
        }
        double[] classifierWeights = new double[pool.length];
        for (int j = 0; j < pool.length; j++) {
            classifierWeights[j] = 1 - weightedErrors[j];
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
