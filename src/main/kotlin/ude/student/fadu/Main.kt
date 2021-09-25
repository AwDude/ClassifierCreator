package ude.student.fadu

import ude.student.fadu.evaluation.ClassifierEvaluation
import weka.classifiers.bayes.NaiveBayes

fun main(@Suppress("UNUSED_PARAMETER") args: Array<String>) {

    ClassifierEvaluation().apply {
        trainAndExportModel(NaiveBayes(), "naiveBayes")
/*        printNaiveBayesProbabilities()
        compareAll()
        showDecisionTreeC45()*/
    }

}