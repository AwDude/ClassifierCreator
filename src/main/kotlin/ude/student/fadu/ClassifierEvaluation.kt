package ude.student.fadu

import weka.classifiers.Evaluation
import weka.core.converters.ConverterUtils
import java.text.NumberFormat
import java.util.*
import kotlin.system.measureTimeMillis

private const val METRIC_SPACE = 9
private const val CLASSIFIER_SPACE = 16
private const val FOLDS = 10
private const val DATASET_FILE_PATH = "data_categorical.arff"
private const val FRACTION_DIGITS = 4

class ClassifierEvaluation {

    private val classifierSelection = //Classifier.values()
        listOf(Classifier.KNN_AUTO_SELECT_K, Classifier.C4_5, Classifier.RANDOM_FOREST, Classifier.NAIVE_BAYES)

    private val formatter = NumberFormat.getNumberInstance(Locale.ENGLISH).apply {
        isGroupingUsed = false
        minimumFractionDigits = FRACTION_DIGITS
        maximumFractionDigits = FRACTION_DIGITS
    }
    private val dataset = initDataSet()

    fun compareAll() {
        println("\n===== Metric comparison of cross validated classifiers =====")
        printClassifierDescriptions()
        printGeneralInfo()
        printEvaluation()
    }

    private fun initDataSet() = ConverterUtils.DataSource(DATASET_FILE_PATH).dataSet.apply {
        if (classIndex() == -1) setClassIndex(numAttributes() - 1)
    }

    private fun printClassifierDescriptions() {
        val text = StringBuilder("\n${"%-${CLASSIFIER_SPACE}s".format("CLASSIFIER")} | DESCRIPTION\n")
        repeat(CLASSIFIER_SPACE) { text.append("-") }
        text.append("-+-----------------------------------------------------------------------------------------------")
        classifierSelection.forEach {
            text.append("\n${"%-${CLASSIFIER_SPACE}s".format(it.title)} | ${it.description}")
        }
        println(text)
    }

    private fun printGeneralInfo() {
        println("\nCross Validation Folds: $FOLDS\nTotal Number of Instances: ${dataset.numInstances()}")
    }

    private fun printEvaluation() {
        printEvaluationMatrixHeader()
        printEvaluationMatrixLine()
        val scores = printEvaluationMatrixEntries()
        printMetrics(scores)
    }

    private fun printEvaluationMatrixHeader() {
        val text = StringBuilder("\n")
        text.append("%-${CLASSIFIER_SPACE}s".format("CLASSIFIER"))
        Metric.values().forEach {
            text.append(" | ${"%${METRIC_SPACE}s".format(it.title)}")
        }
        text.append(" | ${"%${METRIC_SPACE}s".format("Seconds")}")
        println(text)
    }

    private fun printEvaluationMatrixLine() {
        val text = StringBuilder("-")
        repeat(CLASSIFIER_SPACE) { text.append("-") }
        repeat(Metric.values().count() + 1) {
            text.append("+--")
            repeat(METRIC_SPACE) { text.append("-") }
        }
        println(text)
    }

    private fun printEvaluationMatrixEntries() = MetricScores().also { scores ->
        classifierSelection.forEach { classifier ->
            print("%-${CLASSIFIER_SPACE}s".format(classifier.title))
            try {
                val evaluation = Evaluation(dataset)
                val time = measureTimeMillis {
                    evaluation.crossValidateModel(classifier.model, dataset, FOLDS, Random(1))
                } / 1000.0

                Metric.values().forEach { metric ->
                    val result = metric.retrieveFrom(evaluation)
                    scores.add(metric, classifier.title, result)
                    val formatted = format(result)
                    print(" | ${"%${METRIC_SPACE}s".format(formatted)}")
                }
                print(" | ${"%${METRIC_SPACE}s".format(format(time))}")
            } catch (e: Exception) {
                print(" | ERROR")
            }
            println()
        }
    }

    private fun printMetrics(scores: MetricScores, topCount: Int = 5) {
        val scoreSpace = 3 + METRIC_SPACE + CLASSIFIER_SPACE
        val text = StringBuilder("\n")
        text.append("%-${METRIC_SPACE}s".format("METRIC"))
        repeat(topCount) { topIndex ->
            text.append(" | ")
            text.append("%-${scoreSpace}s".format("TOP ${topIndex + 1}"))
        }
        text.append(" | DESCRIPTION\n")
        repeat(METRIC_SPACE) { text.append("-") }
        repeat(topCount) {
            text.append("-+-")
            repeat(scoreSpace) { text.append("-") }
        }
        text.append("-+--------------------------------------------------------------------------\n")

        Metric.values().forEach { metric ->
            text.append("%-${METRIC_SPACE}s".format(metric.title))
            repeat(topCount) { topIndex ->
                text.append(" | ")
                val score = scores.getTop(metric, topIndex)
                val scoreText = score?.let { format(score.score) } ?: ""
                text.append("%${scoreSpace}s".format("${score?.name} ($scoreText)"))
            }
            text.append(" | ${metric.description}\n")
        }
        print(text)
    }

    private fun format(number: Double) =
        if (number % 1.0 == 0.0) number.toInt().toString() else formatter.format(number)
}