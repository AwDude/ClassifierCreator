package ude.student.fadu

import weka.classifiers.Evaluation
import weka.core.converters.ConverterUtils
import java.text.NumberFormat
import java.util.*

private const val METRIC_SPACE = 9
private const val CLASSIFIER_SPACE = 16
private const val FOLDS = 10
private const val DATASET_FILE_PATH = "data_categorical.arff"

class ClassifierEvaluation {

    private val formatter = NumberFormat.getNumberInstance(Locale.ENGLISH).apply {
        isGroupingUsed = false
        minimumFractionDigits = 0
        maximumFractionDigits = 4
    }
    private val dataset = initDataSet()

    fun compareAll() {
        println("\n===== Metric comparison of cross validated classifiers =====")
        printMetricDescriptions()
        printClassifierDescriptions()
        printGeneralInfo()
        printEvaluationMatrixHeader()
        printEvaluationMatrixEntries()
    }

    private fun initDataSet() = ConverterUtils.DataSource(DATASET_FILE_PATH).dataSet.apply {
        if (classIndex() == -1) setClassIndex(numAttributes() - 1)
    }

    private fun printMetricDescriptions() {
        val text = StringBuilder()
        text.append("\n${String.format("%-${METRIC_SPACE}s", "METRIC")} | DESCRIPTION\n")
        repeat(METRIC_SPACE) { text.append("-") }
        text.append("-+------------------------------------------------------------------------------------------------------------")
        Metric.values().forEach {
            text.append("\n${String.format("%-${METRIC_SPACE}s", it.title)} | ${it.description}")
        }
        println(text)
    }

    private fun printClassifierDescriptions() {
        val text = StringBuilder()
        text.append("\n${String.format("%-${CLASSIFIER_SPACE}s", "CLASSIFIER")} | DESCRIPTION\n")
        repeat(CLASSIFIER_SPACE) { text.append("-") }
        text.append("-+------------------------------------------------------------------------------------------------------------\n")
        Classifier.values().forEach {
            text.append("${String.format("%-${CLASSIFIER_SPACE}s", it.title)} | ${it.description}\n")
        }
        println(text)
    }

    private fun printGeneralInfo() {
        println("Cross Validation Folds: $FOLDS\nTotal Number of Instances: ${dataset.numInstances()}\n")
    }

    private fun printEvaluationMatrixHeader() {
        val text = StringBuilder()
        text.append(String.format("%-${CLASSIFIER_SPACE}s", "CLASSIFIER"))
        Metric.values().forEach {
            text.append(" | ${String.format("%-${METRIC_SPACE}s", it.title)}")
        }
        text.append("\n-")
        repeat(CLASSIFIER_SPACE) { text.append("-") }
        repeat(Metric.values().count()) {
            text.append("+--")
            repeat(METRIC_SPACE) { text.append("-") }
        }
        println(text)
    }

    private fun printEvaluationMatrixEntries() = Classifier.values().forEach { classifier ->
        print(String.format("%-${CLASSIFIER_SPACE}s", classifier.title))
        try {
            val evaluation = Evaluation(dataset)
            evaluation.crossValidateModel(classifier.model, dataset, FOLDS, Random(1))

            Metric.values().forEach { metric ->
                val formatted = formatter.format(metric.retrieveFrom(evaluation))
                print(" | ${String.format("%-${METRIC_SPACE}s", formatted)}")
            }
        } catch (e: Exception) {
            print(" | ERROR")
        }
        println()
    }

}