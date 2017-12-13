# jMCS - Java Multiple Classifiers Systems

Java Multiple Classifiers Systems is a Java library for building Multiple
Classifiers Systems (MCSs) on top of
[WEKA](https://www.cs.waikato.ac.nz/ml/weka/).

Since jMCS is heavily dependent on WEKA, it follows
[WEKA's Java requirement](https://www.cs.waikato.ac.nz/ml/weka/requirements.html).
You can check jMCS's Java requirements, and which version of WEKA is needed, in
the table below.

| jMCS | WEKA | Java |
| --- | --- | --- |
| < 1.0 | 3.8.1 | 1.8 |

## What's available?

Currently, jMCS has the following algorithms available to you:

* **Classifier Fusion**:
  * Weighted Voting;
  * Dynamic Voting.
* **Dynamic Selection**:
  * Overall Local Accuracy (OLA);
  * Local Class Accuracy (LCA);
  * KNORA-E;
  * Multi-label meta-learner;
  * Dynamic Voting with Selection (DVS);
  * Multiple Classifier Behavior based selection;
  * Dynamic Selection (DS);

## Building

To build the project, aside from the Java version dependency, you must have
[Maven](https://maven.apache.org/) installed in your machine. Once those are
installed, you can build the project using maven:

```
$ maven clean compile package
```

Once that finishes executing, you will have three JAR files:

1. The library jar file;
2. The tuning experiment executable jar file;
3. The comparison experiment executable jar file.

## Experiments

There are two experiments ready to be executed as part of the school project:
the tuning experiment, which will give the accuracy information on the
multi-label based classifier selection's different threshold configurations; and
the comparison experiment, which outputs the accuracy information for all the
implemented dynamic selection methods.

They are both available as a JAR file, built by Maven. Their usage are as
follows:

```
$ java -jar jmcs-<VERSION>-tuning.jar <MLKNN|CLR> <PATH-TO-DATABASE-FOLDER>
```

and

```
$ java -jar jmcs-<VERSION>-comparison.jar <PATH-TO-DATABASE-FOLDER>
```

They both receive a folder path as the last argument, which should be a valid
folder containing **only** .arff files, with the data sets on which you wish
to test the implemented solutions.
