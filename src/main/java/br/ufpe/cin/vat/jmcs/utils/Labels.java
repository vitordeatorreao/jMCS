package br.ufpe.cin.vat.jmcs.utils;

public final class Labels
{
    public static boolean Equals(double label1, double label2)
    {
        return Math.abs(label1 - label2) < 0.0001;
    }
}
