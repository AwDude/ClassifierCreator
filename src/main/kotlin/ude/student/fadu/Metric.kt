package ude.student.fadu

import weka.classifiers.Evaluation

enum class Metric(val title: String, val description: String) {
    TP_R("TP Rate", "True Positive Rate. Same as Precision, F-Measure and Recall in weighted class average.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedTruePositiveRate()
    },
    FP_R("FP Rate", "False Positive Rate.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedFalsePositiveRate()
    },
    PRECISION("Precision", "Precision. Same as TP Rate, F-Measure and Recall in weighted class average.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedPrecision()
    },
    RECALL("Recall", "Recall. Same as TP Rate, F-Measure and Precision in weighted class average.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedRecall()
    },
    F_MEASURE("F-Measure", "F-Measure. For testing Accuracy. Same as TP Rate, Recall and Precision in weighted class average.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedFMeasure()
    },
    MCC("MCC", "MatthewsCorrelation.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedMatthewsCorrelation()
    },
    ROC_A("ROC Area", "Area under ROC.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedAreaUnderROC()
    },
    PRC_A("PRC Area", "Area under PRC.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedAreaUnderPRC()
    };

    abstract fun retrieveFrom(evaluation: Evaluation): Double
}