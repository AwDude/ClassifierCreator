package ude.student.fadu

import weka.classifiers.Evaluation
import weka.classifiers.trees.*
import weka.core.converters.ConverterUtils
import java.util.*

class ClassifierEvaluation {

    private val folds = 10
    private val text = StringBuilder()
    private val dataset = initDataSet()
    private val classifiers =
        listOf(
            J48(),
            DecisionStump(),
            HoeffdingTree(),
            LMT(),
            RandomForest(),
            RandomTree(),
            REPTree()
        )

    fun compareAll() {
        text.setLength(0) // clear
        text.append("\n===== Metric comparison of cross validated classifiers (weighted class average, $folds folds) =====\n")
        addMetricDescriptions()
        addEvaluationMatrixHeader()
        addEvaluationMatrixEntries()
        print(text.append("\n\n"))
    }

    private fun initDataSet() = ConverterUtils.DataSource("data_categorical.arff").dataSet.apply {
        if (classIndex() == -1) setClassIndex(numAttributes() - 1)
    }

    private fun addMetricDescriptions() {
        text.append("\nMETRIC     | DESCRIPTION\n")
        text.append("-----------+------------------------------------------------------------------------------------------------------------\n")
        Metric.values().forEach {
            text.append("${String.format("%-10s", it.title)} | ${it.description}\n")
        }
    }

    private fun addEvaluationMatrixHeader() {
        text.append("\nCLASSIFIER     ")
        Metric.values().forEach {
            text.append(" | ${String.format("%-10s", it.title)}")
        }
        text.append("\n---------------")
        repeat(Metric.values().count()) {
            text.append("-+-----------")
        }
    }

    private fun addEvaluationMatrixEntries() = classifiers.forEach { classifier ->
        text.append("\n${String.format("%-15s", classifier::class.java.simpleName)}")

        val evaluation = Evaluation(dataset)
        evaluation.crossValidateModel(classifier, dataset, folds, Random(1))

        Metric.values().forEach { metric ->
            text.append(" | ${String.format("%-10.3f", metric.retrieveFrom(evaluation))}")
        }

        //text.append(String.format("%-10.3f", evaluation.weightedFMeasure()))
    }

    private fun evaluate() {
        // number of folds (splits) used for cross validation
        val folds = 10

        // get the dataset by using the previously defined function
        val dataSet = DecisionTreeDemo.initDataSet()

        // instantiate the classifier model. J48 refers to the C4.5 DTC induction algorithm
        val classifier = J48()

        /* create an evaluation object on the given dataset used for executing the validation and storing the results */
        val evaluation = Evaluation(dataSet)

        // initiate the cross validation
        evaluation.crossValidateModel(classifier, dataSet, folds, Random(1))


        /* print the averaged evaluation results for the cross validation over all folds */
        println(evaluation.toSummaryString())
        println(evaluation.toClassDetailsString())


        text.append("                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area\n")
    }

    private fun appendEvaluationText(text: StringBuffer, evaluation: Evaluation) {
        text.append("                 ")
        text.append(String.format("%-9.3f", evaluation.weightedTruePositiveRate()))
        text.append(String.format("%-9.3f", evaluation.weightedFalsePositiveRate()))
        text.append(String.format("%-11.3f", evaluation.weightedPrecision()))
        text.append(String.format("%-9.3f", evaluation.weightedRecall()))
        text.append(String.format("%-10.3f", evaluation.weightedFMeasure()))
        text.append(String.format("%-9.3f", evaluation.weightedMatthewsCorrelation()))
        text.append(String.format("%-10.3f", evaluation.weightedAreaUnderROC()))
        text.append(String.format("%-10.3f", evaluation.weightedAreaUnderPRC()))
    }

}