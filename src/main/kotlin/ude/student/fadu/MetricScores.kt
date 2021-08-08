package ude.student.fadu

class MetricScores {
    private var isSorted = false
    private val metricMap = linkedMapOf<Metric, MutableList<MetricScore>>()

    fun add(metric: Metric, name: String, score: Double) {
        isSorted = false
        val scores = metricMap[metric] ?: mutableListOf<MetricScore>().also { metricMap[metric] = it }
        scores.add(MetricScore(name, score))
    }

    private fun sort() = metricMap.forEach { (metric, scoreList) ->
        if (metric.isBiggerBetter) scoreList.sortDescending() else scoreList.sort()
    }.run { isSorted = true }

    fun getTop(metric: Metric, rank: Int = 0): MetricScore? {
        if (!isSorted) sort()
        return metricMap[metric]?.elementAtOrNull(rank)
    }
}