package ude.student.fadu.evaluation

import weka.classifiers.Classifier
import weka.classifiers.Evaluation
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.trees.J48
import weka.core.Drawable
import weka.core.Instances
import weka.core.SerializationHelper
import weka.core.converters.ConverterUtils
import weka.estimators.NormalEstimator
import weka.gui.treevisualizer.PlaceNode2
import weka.gui.treevisualizer.TreeVisualizer
import java.text.NumberFormat
import java.util.*
import javax.swing.JFrame
import javax.swing.SwingUtilities
import kotlin.system.measureTimeMillis

private const val METRIC_SPACE = 12
private const val CLASSIFIER_SPACE = 16
private const val FOLDS = 10
private const val DATASET_FILE_PATH = "data_categorical.arff"
private const val FRACTION_DIGITS = 4
private const val TABLE_SPLITS = 2

class ClassifierEvaluation {

    private lateinit var classifierSelection: List<Classifiers>

    private val formatter = NumberFormat.getNumberInstance(Locale.ENGLISH).apply {
        isGroupingUsed = false
        minimumFractionDigits = FRACTION_DIGITS
        maximumFractionDigits = FRACTION_DIGITS
    }
    private val dataset = initDataSet()

    fun showDecisionTreeC45() {
        val classifier = J48()
        train(classifier)
        visualize(classifier)
    }

    fun trainAndExportModel(classifier: Classifier, fileName: String) {
        train(classifier)
        SerializationHelper.write("$fileName.model", classifier)
    }

    fun exportHeader() {
        val header = Instances(dataset, 0)
        SerializationHelper.write("dataset.header", header)
    }

    fun train(classifier: Classifier) = classifier.apply { buildClassifier(dataset) }

    fun <T : Drawable> visualize(classifier: T) = SwingUtilities.invokeLater {
        try {
            val visualizeTree = TreeVisualizer(null, classifier.graph(), PlaceNode2())
            val jFrame = JFrame("Weka J48 Klassisfikator: Entscheidungsbaum")
            jFrame.setSize(600, 500)
            jFrame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
            jFrame.contentPane.add(visualizeTree)
            jFrame.isVisible = true
            visualizeTree.fitToScreen()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun compareClassifiers(classifiers: List<Classifiers>) {
        classifierSelection = classifiers
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
        val metrics = Metrics.values()
        val chunkSize = (metrics.size / TABLE_SPLITS) + 1
        val chunks = metrics.asIterable().chunked(chunkSize)
        val scores = MetricScores()

        chunks.forEach { chunkOfMetrics ->
            printEvaluationMatrixHeader(chunkOfMetrics)
            printEvaluationMatrixLine(chunkOfMetrics)
            printEvaluationMatrixEntries(chunkOfMetrics, scores)
        }
        printMetrics(scores)
    }

    private fun printEvaluationMatrixHeader(metrics: List<Metrics>) {
        val text = StringBuilder("\n")
        text.append("%-${CLASSIFIER_SPACE}s".format("CLASSIFIER"))
        metrics.forEach {
            text.append(" | ${"%${METRIC_SPACE}s".format(it.title)}")
        }
        //text.append(" | ${"%${METRIC_SPACE}s".format("Seconds")}")
        println(text)
    }

    private fun printEvaluationMatrixLine(metrics: List<Metrics>) {
        val text = StringBuilder("-")
        repeat(CLASSIFIER_SPACE) { text.append("-") }
        repeat(metrics.count()) {
            text.append("+--")
            repeat(METRIC_SPACE) { text.append("-") }
        }
        println(text)
    }

    private fun printEvaluationMatrixEntries(metrics: List<Metrics>, scores: MetricScores) {
        classifierSelection.forEach { classifier ->
            print("%-${CLASSIFIER_SPACE}s".format(classifier.title))
            try {
                val evaluation = Evaluation(dataset)
                val time = measureTimeMillis {
                    evaluation.crossValidateModel(classifier.model, dataset, FOLDS, Random(1))
                } / 1000.0

                metrics.forEach { metric ->
                    val result = metric.retrieveFrom(evaluation)
                    scores.add(metric, classifier.title, result)
                    val formatted = format(result)
                    print(" | ${"%${METRIC_SPACE}s".format(formatted)}")
                }
                //print(" | ${"%${METRIC_SPACE}s".format(format(time))}")
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

        Metrics.values().forEach { metric ->
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

    fun printNaiveBayesProbabilities() {
        val dataSet = initDataSet()
        val classifier = NaiveBayes()
        classifier.buildClassifier(dataSet)

        val eval = Evaluation(dataSet)
        eval.crossValidateModel(classifier, dataSet, FOLDS, Random(1))

        val probabilities = classifier.conditionalEstimators

        val numAttrs = dataSet.numAttributes() - 1

        var count = 2

        (0..1).forEach { classIndex ->
            val classLabel = dataSet.classAttribute().value(classIndex) + "_severity"
            val classProb = classifier.classEstimator.getProbability(classIndex.toDouble())
            println("[Class]: $classLabel")
            println("P($classLabel) = ${formatter.format(classProb)}")

            (0 until numAttrs).forEach { attrIndex ->
                val attr = dataSet.attribute(attrIndex)
                println("   [Attribute]: ${attr.name()}")
                attr.enumerateValues()?.iterator()?.withIndex()?.forEach { (index, attrVal) ->
                    val attrProb = probabilities[attrIndex][classIndex].getProbability(index.toDouble())
                    println("      P($attrVal|$classLabel) = ${formatter.format(attrProb)}")
                    count++
                } ?: run {
                    val estimator = probabilities[attrIndex][classIndex] as NormalEstimator
                    println("      P( * |$classLabel) = ${estimator.mean}")
                    println("      * Mean conditional probability over all class determined by applying intervals of size ${estimator.precision} on the numeric values.")
                }
            }
            println()
        }

        println("count: $count")
    }
}