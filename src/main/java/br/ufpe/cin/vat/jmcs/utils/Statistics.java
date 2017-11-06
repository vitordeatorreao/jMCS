/* Statistics.java - calculates simple measures about given observations
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

package br.ufpe.cin.vat.jmcs.utils;

/**
 * Provides simple functions to calculate statistics about given samples.
 * @author vitordeatorreao
 * @since 0.1
 */
public final class Statistics
{
    /**
     * Calculates the summation of the observations in the sample
     * @param sample - set of observations collected
     * @return Sum of the observations in the sample
     * @since 0.1
     */
    public static strictfp double Sum(double[] sample)
    {
        if (sample == null)
        {
            throw new IllegalArgumentException(
                "The given array must not be null");
        }
        double sum = 0d;
        for (double observation : sample) sum += observation;
        return sum;
    }

    /**
     * Calculates the sample mean for the given observations
     * @param sample - set of observations collected
     * @return The sample mean for the given observations
     * @since 0.1
     */
    public static strictfp double Mean(double[] sample)
    {
        if (sample == null || sample.length < 1)
        {
            throw new IllegalArgumentException(
                    "The given array must have at least one element");
        }
        return Sum(sample) / sample.length;
    }

    /**
     * Calculates the sample variance for the given observations
     * @param sample - set of observations collected
     * @return The sample variance for the given observations
     * @since 0.1
     */
    public static strictfp double Variance(double[] sample)
    {
        if (sample == null || sample.length < 2)
        {
            throw new IllegalArgumentException(
                    "The given array must have at least two elements");
        }
        double mean = Mean(sample);
        double sumOfDeviations = 0d;
        for (double observation : sample)
        {
            double meanDiff = observation - mean;
            sumOfDeviations += meanDiff * meanDiff;
        }
        return sumOfDeviations / (sample.length - 1);
    }

    /**
     * Calculates the sample standard deviation for the given observations 
     * @param sample - set of observations collected
     * @return The sample standard deviation for the given observations
     * @since 0.1
     */
    public static strictfp double StandardDeviation(double[] sample)
    {
        return Math.sqrt(Variance(sample));
    }
}
