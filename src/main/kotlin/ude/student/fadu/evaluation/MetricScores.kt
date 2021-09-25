package ude.student.fadu.evaluation

class MetricScores {
    private var isSorted = false
    private val metricMap = linkedMapOf<Metrics, MutableList<MetricScore>>()

    fun add(metric: Metrics, name: String, score: Double) {
        isSorted = false
        val scores = metricMap[metric] ?: mutableListOf<MetricScore>().also { metricMap[metric] = it }
        scores.add(MetricScore(name, score))
    }

    private fun sort() = metricMap.forEach { (metric, scoreList) ->
        if (metric.isBiggerBetter) scoreList.sortDescending() else scoreList.sort()
    }.run { isSorted = true }

    fun getTop(metric: Metrics, rank: Int = 0): MetricScore? {
        if (!isSorted) sort()
        return metricMap[metric]?.elementAtOrNull(rank)
    }
}