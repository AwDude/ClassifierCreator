package ude.student.fadu

import ude.student.fadu.evaluation.ClassifierEvaluation
import ude.student.fadu.evaluation.Classifiers

fun main(@Suppress("UNUSED_PARAMETER") args: Array<String>) {

    val allClassifiers = Classifiers.values().toList()
    val classifierSelection =
        listOf(Classifiers.KNN_AUTO_SELECT_K, Classifiers.C4_5, Classifiers.RANDOM_FOREST, Classifiers.NAIVE_BAYES)

    ClassifierEvaluation().apply {

        // showDecisionTreeC45()

        // trainAndExportModel(NaiveBayes(), "naiveBayes")

        // printNaiveBayesProbabilities()

        compareClassifiers(classifierSelection)

        // compareClassifiers(allClassifiers)

    }

}