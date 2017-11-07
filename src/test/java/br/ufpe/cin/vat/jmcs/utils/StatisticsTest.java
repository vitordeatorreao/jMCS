package br.ufpe.cin.vat.jmcs.utils;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class StatisticsTest
{
    @Test(expected = IllegalArgumentException.class)
    public void testSumForNullSample()
    {
        Statistics.Sum(null);
    }

    @Test
    public void testSumForValidInputs()
    {
        // Check if the sum of empty sample is 0.0
        assertEquals(0.0d, Statistics.Sum(new double[0]), 0.0001);
        // check if the sum of a single observation is itself
        double[] sample = { 0.5d } ;
        assertEquals(0.5d, Statistics.Sum(sample), 0.0001);
        // check the sum for multiple observations
        double[] sample2 = { 0.5d, 1d, 0.3d };
        assertEquals(1.8d, Statistics.Sum(sample2), 0.0001);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMeanForNullSample()
    {
        Statistics.Mean(null);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMeanForEmptySample()
    {
        Statistics.Mean(new double[0]);
    }

    @Test
    public void testMeanForValidInputs()
    {
        double[] sample = { 0.5d };
        assertEquals(0.5d, Statistics.Mean(sample), 0.0001);
        double[] sample2 = { 1d, 2d, 3d };
        assertEquals(2.0d,  Statistics.Mean(sample2), 0.0001);
    }
}
